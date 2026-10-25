import Foundation

/// Mistral Voxtral real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.mistral.ai/v1/audio/transcriptions/realtime`.
/// Sends audio as base64-encoded JSON. Accumulates text from `transcription.text.delta` events.
public final class MistralStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var accumulatedText = ""
    private var finalizationContinuation: AsyncStream<String>.Continuation?
    public let finalizationEvents: AsyncStream<String>

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    public init() {
        var finalContinuation: AsyncStream<String>.Continuation!
        finalizationEvents = AsyncStream { finalContinuation = $0 }
        finalizationContinuation = finalContinuation
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
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
        try validateAPIKey(apiKey)
        var components = URLComponents(string: "wss://api.mistral.ai/v1/audio/transcriptions/realtime")!
        components.queryItems = [
            URLQueryItem(name: "model", value: model)
        ]

        guard let url = components.url else {
            throw LLMKitError.invalidURL("wss://api.mistral.ai/v1/audio/transcriptions/realtime")
        }

        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        do {
            try await expectSessionEvent("session.created", from: task)
            // Voxtral 3 locks settings when the first audio chunk arrives.
            try await sendSessionUpdate(language: language, customVocabulary: customVocabulary)
            try await expectSessionEvent("session.updated", from: task)
            eventsContinuation?.yield(.sessionStarted)
        } catch {
            await disconnect()
            throw error
        }

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Mistral streaming.")
        }

        let base64Audio = data.base64EncodedString()
        let message: [String: Any] = [
            "type": "input_audio.append",
            "audio": base64Audio
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: message)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Mistral streaming.")
        }

        let message: [String: Any] = ["type": "input_audio.end"]
        let jsonData = try JSONSerialization.data(withJSONObject: message)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    public func disconnect() async {
        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        finalizationContinuation?.finish()
        accumulatedText = ""
    }

    // MARK: - Private

    private func expectSessionEvent(_ expectedType: String, from task: URLSessionWebSocketTask) async throws {
        let message = try await task.receive()
        let data: Data
        switch message {
        case .string(let text):
            data = Data(text.utf8)
        case .data(let bytes):
            data = bytes
        @unknown default:
            throw LLMKitError.networkError("Unexpected Mistral session message.")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else {
            throw LLMKitError.networkError("Invalid Mistral session message.")
        }
        if type == "error" {
            throw LLMKitError.httpError(statusCode: 400, message: extractErrorMessage(from: json))
        }
        guard type == expectedType else {
            throw LLMKitError.networkError("Expected Mistral \(expectedType), received \(type).")
        }
    }

    private func sendSessionUpdate(language: String?, customVocabulary: [String]) async throws {
        guard let task = webSocketTask else { return }

        var settings: [String: Any] = [
            "audio_format": ["encoding": "pcm_s16le", "sample_rate": 16000],
            "target_streaming_delay_ms": 480
        ]
        if let language, !language.isEmpty, language != "auto" {
            settings["language"] = language
        }
        let vocabulary = customVocabulary.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        if !vocabulary.isEmpty {
            settings["context_bias"] = vocabulary
        }
        let message: [String: Any] = [
            "type": "session.update",
            "session": settings
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: message)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    private func receiveLoop() async {
        guard let task = webSocketTask else { return }

        while !Task.isCancelled {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        handleMessage(text)
                    }
                @unknown default:
                    break
                }
            } catch {
                if !Task.isCancelled {
                    eventsContinuation?.yield(.error(error.localizedDescription))
                    finalizationContinuation?.finish()
                }
                break
            }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "transcription.text.delta":
            if let deltaText = json["text"] as? String {
                accumulatedText += deltaText
                eventsContinuation?.yield(.partial(text: accumulatedText))
            }

        case "transcription.done":
            guard let finalText = json["text"] as? String else {
                eventsContinuation?.yield(.error("Mistral final transcription is missing text."))
                finalizationContinuation?.finish()
                return
            }
            eventsContinuation?.yield(.committed(text: finalText))
            finalizationContinuation?.yield(finalText)
            finalizationContinuation?.finish()
            accumulatedText = ""

        case "transcription.language":
            break // Language detection — informational only

        case "session.updated":
            break // Session config acknowledged

        case "error":
            let errorMsg = extractErrorMessage(from: json)
            eventsContinuation?.yield(.error(errorMsg))
            finalizationContinuation?.finish()

        default:
            break
        }
    }

    private func extractErrorMessage(from json: [String: Any]) -> String {
        if let error = json["error"] as? [String: Any] {
            if let message = error["message"] as? String { return message }
            if let detail = error["detail"] as? String { return detail }
        }
        if let error = json["error"] as? String { return error }
        if let message = json["message"] as? String { return message }
        return "Unknown error"
    }
}
