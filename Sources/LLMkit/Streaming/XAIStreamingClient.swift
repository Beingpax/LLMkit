import Foundation

/// xAI real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.x.ai/v1/stt`.
/// Sends raw binary PCM audio (signed 16-bit little-endian). Configuration via URL query params.
/// API docs: https://docs.x.ai/developers/model-capabilities/audio/speech-to-text
public final class XAIStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

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

    /// Connects to the xAI streaming endpoint.
    ///
    /// The `model` parameter is accepted for protocol conformance but currently ignored —
    /// the xAI STT endpoint does not expose per-model selection.
    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        var components = URLComponents(string: "wss://api.x.ai/v1/stt")!
        var queryItems: [URLQueryItem] = [
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "encoding", value: "pcm"),
            URLQueryItem(name: "interim_results", value: "true"),
        ]

        if let language, language != "auto", !language.isEmpty {
            queryItems.append(URLQueryItem(name: "language", value: language))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw LLMKitError.invalidURL("wss://api.x.ai/v1/stt")
        }

        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        // Wait for `transcript.created` handshake before returning.
        let message = try await task.receive()
        switch message {
        case .string(let text):
            if let data = text.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let type = json["type"] as? String {
                if type == "transcript.created" {
                    eventsContinuation?.yield(.sessionStarted)
                } else if type == "error" {
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
            throw LLMKitError.networkError("Not connected to xAI streaming.")
        }
        try await task.send(.data(data))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to xAI streaming.")
        }

        let endMessage: [String: Any] = ["type": "audio.done"]
        let jsonData = try JSONSerialization.data(withJSONObject: endMessage)
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
              let type = json["type"] as? String else { return }

        switch type {
        case "transcript.partial":
            guard let transcript = json["text"] as? String,
                  !transcript.trimmingCharacters(in: .whitespaces).isEmpty else { return }
            eventsContinuation?.yield(.partial(text: transcript))

        case "transcript.done":
            guard let transcript = json["text"] as? String else { return }
            eventsContinuation?.yield(.committed(text: transcript))

        case "error":
            let message = json["message"] as? String ?? "xAI streaming error"
            eventsContinuation?.yield(.error(message))

        default:
            break
        }
    }
}
