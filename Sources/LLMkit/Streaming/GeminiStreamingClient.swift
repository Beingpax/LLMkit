import Foundation

/// Real-time transcription client for `gemini-3.5-transcribe-live`.
///
/// Sends base64-encoded raw PCM chunks over Gemini's Live API WebSocket and
/// emits speculative interim text plus authoritative finalized segments.
public final class GeminiStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {
    private static let endpoint =
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    private let endpointURL: URL
    private let stateLock = NSLock()
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var receiveTask: Task<Void, Never>?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    private var setupComplete = false
    private var setupError: String?
    private var finalizationRequested = false
    private var receivedFinalAfterCommit = false
    private var generationCompletedAfterCommit = false
    private var audioEnded = false
    private var sentAudio = false
    private var sessionError: String?
    private var disconnecting = false
    private var lifecycle: GeminiWebSocketLifecycle?
    private var accumulatedFinalText = ""

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    public private(set) var finalizationEvents: AsyncStream<String>

    public convenience init() {
        self.init(endpointURL: URL(string: Self.endpoint)!)
    }

    init(endpointURL: URL) {
        self.endpointURL = endpointURL
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation

        var finalizationContinuation: AsyncStream<String>.Continuation!
        finalizationEvents = AsyncStream { finalizationContinuation = $0 }
        self.finalizationContinuation = finalizationContinuation
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
    }

    public func connect(
        apiKey: String,
        model: String,
        language: String?,
        customVocabulary: [String] = []
    ) async throws {
        try validateAPIKey(apiKey)
        let requestedModel = model.lowercased()
        guard requestedModel == "gemini-3.5-transcribe"
                || requestedModel == "gemini-3.5-transcribe-live" else {
            throw LLMKitError.unsupportedModel(model)
        }

        guard var components = URLComponents(url: endpointURL, resolvingAgainstBaseURL: false) else {
            throw LLMKitError.invalidURL(Self.endpoint)
        }
        components.queryItems = (components.queryItems ?? []) + [URLQueryItem(name: "key", value: apiKey)]
        guard let endpoint = components.url else {
            throw LLMKitError.invalidURL(Self.endpoint)
        }

        stateLock.withLock {
            setupComplete = false
            setupError = nil
            finalizationRequested = false
            receivedFinalAfterCommit = false
            generationCompletedAfterCommit = false
            audioEnded = false
            sentAudio = false
            sessionError = nil
            disconnecting = false
            accumulatedFinalText = ""
        }

        let lifecycle = GeminiWebSocketLifecycle()
        let session = URLSession(configuration: .ephemeral, delegate: lifecycle, delegateQueue: nil)
        let task = session.webSocketTask(with: endpoint)
        stateLock.withLock {
            self.lifecycle = lifecycle
            urlSession = session
            webSocketTask = task
        }
        task.resume()

        let receiver = Task<Void, Never> { [weak self] in
            await self?.receiveLoop()
        }
        stateLock.withLock {
            receiveTask = receiver
        }

        let liveModel = "gemini-3.5-transcribe-live"
        let languageCodes: [String]
        if let language, !language.isEmpty, language.lowercased() != "auto" {
            languageCodes = [language]
        } else {
            languageCodes = []
        }

        let setup = GeminiLiveSetupMessage(
            setup: GeminiLiveSetup(
                model: "models/\(liveModel)",
                generationConfig: GeminiLiveGenerationConfig(responseModalities: ["TEXT"]),
                inputAudioTranscription: GeminiLiveTranscriptionConfig(
                    languageCodes: languageCodes,
                    customVocabulary: Self.normalizedVocabulary(customVocabulary),
                    mode: "VERBATIM"
                )
            )
        )
        do {
            try await sendJSON(setup)
            try await waitForSetup()
        } catch {
            await disconnect()
            throw error
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard !data.isEmpty else { return }
        try stateLock.withLock {
            try requireStreaming()
            guard !audioEnded else {
                throw LLMKitError.networkError("Gemini audio input has already ended.")
            }
            sentAudio = true
        }
        let message = GeminiLiveAudioMessage(
            realtimeInput: GeminiLiveRealtimeInput(
                audio: GeminiLiveAudio(
                    data: data.base64EncodedString(),
                    mimeType: "audio/pcm;rate=16000"
                ),
                audioStreamEnd: nil
            )
        )
        do {
            try await sendJSON(message)
        } catch {
            failSession(error.localizedDescription)
            throw error
        }
    }

    public func commit() async throws {
        let shouldCommit = try stateLock.withLock {
            try requireStreaming()
            guard !audioEnded else { return false }
            audioEnded = true
            return true
        }
        guard shouldCommit else { return }

        let message = GeminiLiveAudioMessage(
            realtimeInput: GeminiLiveRealtimeInput(audio: nil, audioStreamEnd: true)
        )
        do {
            try await sendJSON(message)
        } catch {
            failSession(error.localizedDescription)
            throw error
        }

        try stateLock.withLock {
            try requireStreaming()
            finalizationRequested = true
        }

        // An earlier final or the absence of an interim says nothing about audio
        // still being processed. A fresh inputTranscription and generationComplete
        // are both required; either can arrive first.
        // If Gemini has already endpointed and sends no further result,
        // the caller must time out and use its complete recording as a fallback.
        let emptyInput = stateLock.withLock { () -> Bool in
            guard !sentAudio, finalizationRequested, sessionError == nil else { return false }
            finalizationRequested = false
            return true
        }
        if emptyInput {
            finalizationContinuation?.yield("")
        }
    }

    public func disconnect() async {
        let resources = stateLock.withLock { () -> (URLSessionWebSocketTask?, URLSession?, GeminiWebSocketLifecycle?, Task<Void, Never>?) in
            disconnecting = true
            finalizationRequested = false
            let resources = (webSocketTask, urlSession, lifecycle, receiveTask)
            webSocketTask = nil
            urlSession = nil
            lifecycle = nil
            receiveTask = nil
            return resources
        }
        let (socket, session, lifecycle, receiver) = resources
        socket?.cancel(with: .normalClosure, reason: nil)
        // Keep the URLSession alive for the closing handshake. Cancelling the
        // session immediately can tear down the transport before the close frame.
        if let lifecycle, await lifecycle.waitForClose() {
            session?.finishTasksAndInvalidate()
        } else {
            session?.invalidateAndCancel()
        }
        receiver?.cancel()
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
    }

    /// Called only while holding stateLock.
    private func requireStreaming() throws {
        if let sessionError {
            throw LLMKitError.networkError(sessionError)
        }
        guard setupComplete, !disconnecting, webSocketTask != nil else {
            throw LLMKitError.networkError("Not connected to Gemini live transcription.")
        }
    }

    private func failSession(_ message: String) {
        let shouldReport = stateLock.withLock {
            guard !disconnecting, sessionError == nil else { return false }
            sessionError = message
            setupError = message
            finalizationRequested = false
            return true
        }
        guard shouldReport else { return }
        eventsContinuation?.yield(.error(message))
        // Ending this stream wakes callers immediately so they can fall back;
        // previously a dead connection could keep them waiting for ten seconds.
        finalizationContinuation?.finish()
    }

    private func waitForSetup() async throws {
        for _ in 0..<100 {
            let state = stateLock.withLock { (setupComplete, setupError, disconnecting) }
            if state.2 { throw CancellationError() }
            if let error = state.1 {
                throw LLMKitError.networkError(error)
            }
            if state.0 {
                eventsContinuation?.yield(.sessionStarted)
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw LLMKitError.timeout
    }

    private func sendJSON<T: Encodable>(_ value: T) async throws {
        guard let task = stateLock.withLock({ webSocketTask }) else {
            throw LLMKitError.networkError("Not connected to Gemini live transcription.")
        }

        let data: Data
        do {
            data = try JSONEncoder().encode(value)
        } catch {
            throw LLMKitError.encodingError
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw LLMKitError.encodingError
        }
        try await task.send(.string(text))
    }

    private func receiveLoop() async {
        guard let task = stateLock.withLock({ webSocketTask }) else { return }

        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                let data: Data
                switch message {
                case .string(let text):
                    data = Data(text.utf8)
                case .data(let value):
                    data = value
                @unknown default:
                    continue
                }
                handleMessage(data)
            } catch {
                guard !Task.isCancelled else { return }
                failSession(error.localizedDescription)
                return
            }
        }
    }

    private func handleMessage(_ data: Data) {
        let response: GeminiLiveResponse
        do {
            response = try JSONDecoder().decode(GeminiLiveResponse.self, from: data)
        } catch {
            failSession("Invalid Gemini live transcription response.")
            return
        }

        if response.setupComplete != nil {
            stateLock.withLock {
                setupComplete = true
            }
        }

        if let message = response.error?.message, !message.isEmpty {
            failSession(message)
            return
        }

        if let interim = response.serverContent?.interimInputTranscription?.text,
           !interim.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            eventsContinuation?.yield(.partial(text: interim))
        }

        if let final = response.serverContent?.inputTranscription?.text {
            stateLock.withLock {
                guard sessionError == nil, !disconnecting else { return }
                let trimmed = final.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    accumulatedFinalText = accumulatedFinalText.isEmpty
                        ? trimmed
                        : accumulatedFinalText + " " + trimmed
                }
                if finalizationRequested {
                    receivedFinalAfterCommit = true
                }
            }
            eventsContinuation?.yield(.committed(text: final))
        }

        let finalizedText = stateLock.withLock { () -> String? in
            if finalizationRequested, response.serverContent?.generationComplete == true {
                generationCompletedAfterCommit = true
            }
            guard finalizationRequested, receivedFinalAfterCommit,
                  generationCompletedAfterCommit, sessionError == nil, !disconnecting else { return nil }
            finalizationRequested = false
            return accumulatedFinalText
        }
        if let finalizedText {
            finalizationContinuation?.yield(finalizedText)
        }
    }

    private static func normalizedVocabulary(_ terms: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []

        for term in terms {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.lowercased()
            guard seen.insert(key).inserted else { continue }
            result.append(trimmed)
            if result.count == 1_000 { break }
        }
        return result
    }
}

private struct GeminiLiveSetupMessage: Encodable, Sendable {
    let setup: GeminiLiveSetup
}

private struct GeminiLiveSetup: Encodable, Sendable {
    let model: String
    let generationConfig: GeminiLiveGenerationConfig
    let inputAudioTranscription: GeminiLiveTranscriptionConfig
}

private struct GeminiLiveGenerationConfig: Encodable, Sendable {
    let responseModalities: [String]
}

private struct GeminiLiveTranscriptionConfig: Encodable, Sendable {
    let languageCodes: [String]
    let customVocabulary: [String]
    let mode: String
}

private struct GeminiLiveAudioMessage: Encodable, Sendable {
    let realtimeInput: GeminiLiveRealtimeInput
}

private struct GeminiLiveRealtimeInput: Encodable, Sendable {
    let audio: GeminiLiveAudio?
    let audioStreamEnd: Bool?
}

private struct GeminiLiveAudio: Encodable, Sendable {
    let data: String
    let mimeType: String
}

private struct GeminiLiveResponse: Decodable, Sendable {
    let setupComplete: GeminiLiveSetupComplete?
    let serverContent: GeminiLiveServerContent?
    let error: GeminiLiveError?
}

private struct GeminiLiveSetupComplete: Decodable, Sendable {}

private struct GeminiLiveServerContent: Decodable, Sendable {
    let interimInputTranscription: GeminiLiveTranscript?
    let inputTranscription: GeminiLiveTranscript?
    let generationComplete: Bool?
}

private struct GeminiLiveTranscript: Decodable, Sendable {
    let text: String
}

private struct GeminiLiveError: Decodable, Sendable {
    let message: String
}

/// Observes actual task completion rather than cancelling the transport at close.
private final class GeminiWebSocketLifecycle: NSObject, URLSessionWebSocketDelegate, @unchecked Sendable {
    private let closeEvents: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    override init() {
        (closeEvents, continuation) = AsyncStream.makeStream(of: Void.self)
        super.init()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        continuation.finish()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        continuation.finish()
    }

    func waitForClose() async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask { [closeEvents] in
                for await _ in closeEvents { }
                return !Task.isCancelled
            }
            group.addTask {
                try? await Task.sleep(for: .seconds(2))
                return false
            }
            let result = await group.next() ?? false
            group.cancelAll()
            return result
        }
    }
}
