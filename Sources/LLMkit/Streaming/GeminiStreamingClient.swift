import Foundation

/// Real-time transcription client for `gemini-3.5-transcribe-live`.
///
/// Sends base64-encoded raw PCM chunks over Gemini's Live API WebSocket and
/// emits speculative interim text plus authoritative finalized segments.
public final class GeminiStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {
    private static let endpoint =
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

    private let stateLock = NSLock()
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var receiveTask: Task<Void, Never>?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    private var setupComplete = false
    private var setupError: String?
    private var finalizationRequested = false
    private var hasOutstandingInterim = false
    private var accumulatedFinalText = ""

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    public private(set) var finalizationEvents: AsyncStream<String>

    public init() {
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

        guard var components = URLComponents(string: Self.endpoint) else {
            throw LLMKitError.invalidURL(Self.endpoint)
        }
        components.queryItems = [URLQueryItem(name: "key", value: apiKey)]
        guard let endpoint = components.url else {
            throw LLMKitError.invalidURL(Self.endpoint)
        }

        stateLock.withLock {
            setupComplete = false
            setupError = nil
            finalizationRequested = false
            hasOutstandingInterim = false
            accumulatedFinalText = ""
        }

        let session = URLSession(configuration: .ephemeral)
        let task = session.webSocketTask(with: endpoint)
        urlSession = session
        webSocketTask = task
        task.resume()

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }

        let requestedModel = model.lowercased()
        guard requestedModel == "gemini-3.5-transcribe"
                || requestedModel == "gemini-3.5-transcribe-live" else {
            throw LLMKitError.unsupportedModel(model)
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
        try await sendJSON(setup)
        try await waitForSetup()
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard !data.isEmpty else { return }
        let message = GeminiLiveAudioMessage(
            realtimeInput: GeminiLiveRealtimeInput(
                audio: GeminiLiveAudio(
                    data: data.base64EncodedString(),
                    mimeType: "audio/pcm;rate=16000"
                ),
                audioStreamEnd: nil
            )
        )
        try await sendJSON(message)
    }

    public func commit() async throws {
        stateLock.withLock {
            finalizationRequested = true
        }
        let message = GeminiLiveAudioMessage(
            realtimeInput: GeminiLiveRealtimeInput(
                audio: nil,
                audioStreamEnd: true
            )
        )
        try await sendJSON(message)

        // audioStreamEnd flushes only audio Gemini has not already finalized.
        // If automatic endpointing already produced the authoritative final
        // transcript and no newer interim exists, there will be no additional
        // inputTranscription event to wait for.
        let alreadyFinalizedText = stateLock.withLock { () -> String? in
            guard finalizationRequested,
                  !hasOutstandingInterim,
                  !accumulatedFinalText.isEmpty else { return nil }
            finalizationRequested = false
            return accumulatedFinalText
        }
        if let alreadyFinalizedText {
            finalizationContinuation?.yield(alreadyFinalizedText)
        }
    }

    public func disconnect() async {
        stateLock.withLock {
            finalizationRequested = false
        }
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
    }

    private func waitForSetup() async throws {
        for _ in 0..<100 {
            let state = stateLock.withLock { (setupComplete, setupError) }
            if state.0 {
                eventsContinuation?.yield(.sessionStarted)
                return
            }
            if let error = state.1 {
                throw LLMKitError.networkError(error)
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw LLMKitError.timeout
    }

    private func sendJSON<T: Encodable>(_ value: T) async throws {
        guard let task = webSocketTask else {
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
        guard let task = webSocketTask else { return }

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
                let description = error.localizedDescription
                stateLock.withLock {
                    if !setupComplete {
                        setupError = description
                    }
                }
                eventsContinuation?.yield(.error(description))
                return
            }
        }
    }

    private func handleMessage(_ data: Data) {
        guard let response = try? JSONDecoder().decode(GeminiLiveResponse.self, from: data) else {
            return
        }

        if response.setupComplete != nil {
            stateLock.withLock {
                setupComplete = true
            }
        }

        if let message = response.error?.message, !message.isEmpty {
            stateLock.withLock {
                setupError = message
            }
            eventsContinuation?.yield(.error(message))
        }

        if let interim = response.serverContent?.interimInputTranscription?.text,
           !interim.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            stateLock.withLock {
                hasOutstandingInterim = true
            }
            eventsContinuation?.yield(.partial(text: interim))
        }

        if let final = response.serverContent?.inputTranscription?.text {
            let trimmed = final.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                stateLock.withLock {
                    accumulatedFinalText = accumulatedFinalText.isEmpty
                        ? trimmed
                        : accumulatedFinalText + " " + trimmed
                    hasOutstandingInterim = false
                }
            } else {
                stateLock.withLock {
                    hasOutstandingInterim = false
                }
            }
            eventsContinuation?.yield(.committed(text: final))
            // audioStreamEnd asks Gemini to flush the current audio stream. The
            // resulting inputTranscription is the authoritative finalized text.
            let finalText = stateLock.withLock { () -> String? in
                guard finalizationRequested else { return nil }
                finalizationRequested = false
                return accumulatedFinalText
            }
            if let finalText {
                finalizationContinuation?.yield(finalText)
            }
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
}

private struct GeminiLiveTranscript: Decodable, Sendable {
    let text: String
}

private struct GeminiLiveError: Decodable, Sendable {
    let message: String
}
