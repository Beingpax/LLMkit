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

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    public init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
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

        // Wait for session.created handshake
        let message = try await task.receive()
        switch message {
        case .string(let text):
            if let data = text.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "session.created" {
                    eventsContinuation?.yield(.sessionStarted)
                } else if type == "error" {
                    let errorMsg = extractErrorMessage(from: json)
                    throw LLMKitError.httpError(statusCode: 400, message: errorMsg)
                }
            }
        case .data:
            break
        @unknown default:
            break
        }

        // Send session.update with audio format
        try await sendSessionUpdate()

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
        accumulatedText = ""
    }

    // MARK: - Private

    private func sendSessionUpdate() async throws {
        guard let task = webSocketTask else { return }

        let message: [String: Any] = [
            "type": "session.update",
            "session": [
                "audio_format": [
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000
                ]
            ]
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
            let finalText = accumulatedText
            eventsContinuation?.yield(.committed(text: finalText))
            accumulatedText = ""

        case "transcription.language":
            break // Language detection — informational only

        case "session.updated":
            break // Session config acknowledged

        case "error":
            let errorMsg = extractErrorMessage(from: json)
            eventsContinuation?.yield(.error(errorMsg))

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
