import Foundation

/// AssemblyAI realtime transport. Model identifiers are passed directly to the
/// API so new compatible models can be selected without updating this library.
///
/// Connects to `wss://streaming.assemblyai.com/v3/ws` and sends raw PCM16,
/// 16 kHz, mono, little-endian audio frames.
public final class AssemblyAIStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {
    private static let keytermLimit = 100
    private static let minimumChunkBytes = 1_600

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var pendingAudio = Data()
    private var committedTurns: [Int: String] = [:]
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    public let finalizationEvents: AsyncStream<String>
    private var didSendTerminate = false
    private var didReceiveTermination = false

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    public init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        var finalContinuation: AsyncStream<String>.Continuation!
        finalizationEvents = AsyncStream { finalContinuation = $0 }
        finalizationContinuation = finalContinuation
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        try await connect(apiKey: apiKey, model: model, language: language, prompt: nil, customVocabulary: customVocabulary)
    }

    /// Additional connection parameters are forwarded to AssemblyAI, allowing
    /// callers to adopt new API options without waiting for a library release.
    /// They override default parameters, but cannot override model selection or
    /// inject authentication into the URL. JSON-valued options must be encoded
    /// as JSON strings (for example, `language_codes: "[\"en\",\"es\"]"`).
    public func connect(
        apiKey: String,
        model: String,
        language: String?,
        prompt: String?,
        customVocabulary: [String] = [],
        additionalParameters: [String: String] = [:]
    ) async throws {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.missingAPIKey
        }

        guard let url = Self.streamingURL(model: model, language: language, customVocabulary: customVocabulary, prompt: prompt, additionalParameters: additionalParameters) else {
            throw LLMKitError.invalidURL("wss://streaming.assemblyai.com/v3/ws")
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)
        urlSession = session
        webSocketTask = task
        pendingAudio.removeAll(keepingCapacity: true)
        committedTurns.removeAll()
        didSendTerminate = false
        didReceiveTermination = false
        task.resume()

        do {
            try await waitForBeginEvent(from: task)
        } catch {
            task.cancel(with: .normalClosure, reason: nil)
            session.invalidateAndCancel()
            webSocketTask = nil
            urlSession = nil
            throw error
        }

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to AssemblyAI streaming.")
        }

        pendingAudio.append(data)
        while pendingAudio.count >= Self.minimumChunkBytes {
            let chunk = pendingAudio.prefix(Self.minimumChunkBytes)
            try await task.send(.data(Data(chunk)))
            pendingAudio.removeFirst(Self.minimumChunkBytes)
        }
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to AssemblyAI streaming.")
        }

        if !pendingAudio.isEmpty {
            // The API requires at least 50 ms per frame, including the final tail.
            if pendingAudio.count < Self.minimumChunkBytes {
                pendingAudio.append(Data(count: Self.minimumChunkBytes - pendingAudio.count))
            }
            try await task.send(.data(pendingAudio))
            pendingAudio.removeAll(keepingCapacity: true)
        }

        didSendTerminate = true
        try await task.send(.string(#"{"type":"Terminate"}"#))
    }

    public func disconnect() async {
        receiveTask?.cancel()
        receiveTask = nil

        if let task = webSocketTask {
            if !didSendTerminate {
                try? await task.send(.string(#"{"type":"Terminate"}"#))
            }
            task.cancel(with: .normalClosure, reason: nil)
        }

        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
        pendingAudio.removeAll(keepingCapacity: false)
        committedTurns.removeAll()
        didSendTerminate = false
        didReceiveTermination = false
    }

    // MARK: - Private

    static func streamingURL(model: String, language: String?, customVocabulary: [String], prompt: String? = nil, additionalParameters: [String: String] = [:]) -> URL? {
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        var components = URLComponents(string: "wss://streaming.assemblyai.com/v3/ws")
        var queryItems = [
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "encoding", value: "pcm_s16le"),
            URLQueryItem(name: "speech_model", value: model),
            URLQueryItem(name: "mode", value: "balanced")
        ]

        if let language,
           !language.isEmpty,
           language != "auto" {
            queryItems.append(URLQueryItem(name: "language_codes", value: jsonArrayString([language])))
        }

        let keyterms = normalizedKeyterms(customVocabulary)
        if let keytermsJSON = jsonArrayString(keyterms), !keyterms.isEmpty {
            queryItems.append(URLQueryItem(name: "keyterms_prompt", value: keytermsJSON))
        }

        if let prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            queryItems.append(URLQueryItem(name: "prompt", value: prompt))
        }
        let protectedParameters: Set<String> = ["speech_model", "token", "api_key", "authorization"]
        for (name, value) in additionalParameters.sorted(by: { $0.key < $1.key }) {
            guard !protectedParameters.contains(name.lowercased()) else { continue }
            queryItems.removeAll { $0.name == name }
            queryItems.append(URLQueryItem(name: name, value: value))
        }

        components?.queryItems = queryItems
        return components?.url
    }

    private static func normalizedKeyterms(_ customVocabulary: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for term in customVocabulary {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            let wordCount = trimmed.split(separator: " ").count
            guard !trimmed.isEmpty, trimmed.count <= 50, wordCount <= 6 else { continue }
            let key = trimmed.lowercased()
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            result.append(trimmed)
            if result.count == keytermLimit { break }
        }
        return result
    }

    private static func jsonArrayString(_ values: [String]) -> String? {
        guard JSONSerialization.isValidJSONObject(values),
              let data = try? JSONSerialization.data(withJSONObject: values),
              let string = String(data: data, encoding: .utf8) else {
            return nil
        }
        return string
    }

    private func waitForBeginEvent(from task: URLSessionWebSocketTask) async throws {
        do {
            while true {
                let message = try await task.receive()
                let text: String?
                switch message {
                case .string(let value):
                    text = value
                case .data(let data):
                    text = String(data: data, encoding: .utf8)
                @unknown default:
                    text = nil
                }

                guard let text else { continue }
                guard let data = text.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    continue
                }

                if let error = Self.serverError(json) {
                    throw LLMKitError.httpError(statusCode: 400, message: error)
                }

                if json["type"] as? String == "Begin" {
                    eventsContinuation?.yield(.sessionStarted)
                    return
                }

                handleMessage(json)
            }
        } catch let error as LLMKitError {
            throw error
        } catch {
            throw LLMKitError.networkError("Failed to start AssemblyAI streaming session: \(error.localizedDescription)")
        }
    }

    private func receiveLoop() async {
        guard let task = webSocketTask else { return }

        while !Task.isCancelled && !didReceiveTermination {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleTextMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleTextMessage(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if !Task.isCancelled {
                    eventsContinuation?.yield(.error(error.localizedDescription))
                }
                break
            }
        }
    }

    private func handleTextMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        handleMessage(json)
    }

    private static func serverError(_ json: [String: Any]) -> String? {
        if let error = json["error"] as? String { return error }
        if json["type"] as? String == "Error" {
            return json["message"] as? String ?? "AssemblyAI streaming failed."
        }
        return nil
    }

    private func handleMessage(_ json: [String: Any]) {
        if let error = Self.serverError(json) {
            eventsContinuation?.yield(.error(error))
            return
        }

        guard let type = json["type"] as? String else { return }
        switch type {
        case "Turn":
            handleTurn(json)
        case "Termination":
            didReceiveTermination = true
            let text = committedTurns.keys.sorted().compactMap { committedTurns[$0] }.joined(separator: " ")
            finalizationContinuation?.yield(text)
            finalizationContinuation?.finish()
            eventsContinuation?.yield(.committed(text: ""))
        default:
            break
        }
    }

    private func handleTurn(_ json: [String: Any]) {
        let transcript = (json["transcript"] as? String) ?? ""
        guard !transcript.isEmpty else { return }

        let endOfTurn = (json["end_of_turn"] as? Bool) ?? false
        let turnOrder = json["turn_order"] as? Int

        if endOfTurn, let turnOrder {
            let isNewTurn = committedTurns[turnOrder] == nil
            committedTurns[turnOrder] = transcript
            if isNewTurn {
                eventsContinuation?.yield(.committed(text: transcript))
            }
        } else if !endOfTurn {
            eventsContinuation?.yield(.partial(text: transcript))
        }
    }
}
