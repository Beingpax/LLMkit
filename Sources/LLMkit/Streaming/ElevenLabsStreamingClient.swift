import Foundation

/// ElevenLabs Scribe V2 real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.elevenlabs.io/v1/speech-to-text/realtime`.
/// Sends audio as base64-encoded JSON chunks. Uses VAD-based commit strategy.
public final class ElevenLabsStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?

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
        var components = URLComponents(string: "wss://api.elevenlabs.io/v1/speech-to-text/realtime")!
        var queryItems: [URLQueryItem] = [
            URLQueryItem(name: "model_id", value: model),
            URLQueryItem(name: "audio_format", value: "pcm_16000"),
            URLQueryItem(name: "commit_strategy", value: "vad"),
        ]

        if let language, language != "auto", !language.isEmpty {
            queryItems.append(URLQueryItem(name: "language_code", value: language))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw LLMKitError.invalidURL("wss://api.elevenlabs.io/v1/speech-to-text/realtime")
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        // Wait for session_started handshake
        let message = try await task.receive()
        switch message {
        case .string(let text):
            if let data = text.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let messageType = json["message_type"] as? String {
                if messageType == "session_started" {
                    eventsContinuation?.yield(.sessionStarted)
                } else if messageType == "error" || messageType == "auth_error" {
                    let errorMsg = json["message"] as? String ?? "Unknown error"
                    throw LLMKitError.httpError(statusCode: 401, message: errorMsg)
                }
            }
        case .data:
            break
        @unknown default:
            break
        }

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to ElevenLabs streaming.")
        }

        let base64Audio = data.base64EncodedString()
        let message: [String: Any] = [
            "message_type": "input_audio_chunk",
            "audio_base_64": base64Audio,
            "commit": false,
            "sample_rate": 16000
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: message)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to ElevenLabs streaming.")
        }

        let message: [String: Any] = [
            "message_type": "input_audio_chunk",
            "audio_base_64": "",
            "commit": true,
            "sample_rate": 16000
        ]

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
    }

    // MARK: - Private

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
              let messageType = json["message_type"] as? String else { return }

        switch messageType {
        case "partial_transcript":
            if let transcript = json["text"] as? String {
                eventsContinuation?.yield(.partial(text: transcript))
            }

        case "committed_transcript", "committed_transcript_with_timestamps":
            if let transcript = json["text"] as? String {
                eventsContinuation?.yield(.committed(text: transcript))
            }

        case "error", "auth_error", "quota_exceeded", "rate_limited",
             "resource_exhausted", "session_time_limit_exceeded",
             "input_error", "chunk_size_exceeded", "transcriber_error":
            let errorMsg = json["message"] as? String ?? messageType
            eventsContinuation?.yield(.error(errorMsg))

        default:
            break
        }
    }
}
