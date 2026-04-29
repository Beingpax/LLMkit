import Foundation

/// Cartesia Ink real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.cartesia.ai/stt/websocket`.
/// Sends raw binary PCM audio (16-bit signed little-endian, 16 kHz).
/// Use `"finalize"` to flush pending audio; `"done"` for graceful session close.
/// API docs: https://docs.cartesia.ai/api-reference/stt/stt
public final class CartesiaStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

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

    /// Cartesia STT does not support custom vocabulary — the parameter is accepted for protocol
    /// conformance but silently ignored.
    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        var components = URLComponents(string: "wss://api.cartesia.ai/stt/websocket")!

        let lang = (language?.isEmpty == false && language != "auto") ? language! : "en"
        components.queryItems = [
            URLQueryItem(name: "model", value: model),
            URLQueryItem(name: "language", value: lang),
            URLQueryItem(name: "encoding", value: "pcm_s16le"),
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "cartesia_version", value: "2026-03-01"),
        ]

        guard let url = components.url else {
            throw LLMKitError.invalidURL("wss://api.cartesia.ai/stt/websocket")
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)
        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        // Cartesia has no session handshake message — emit sessionStarted immediately after connection.
        eventsContinuation?.yield(.sessionStarted)

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Cartesia streaming.")
        }
        try await task.send(.data(data))
    }

    /// Flushes any buffered audio and triggers transcription of remaining speech.
    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Cartesia streaming.")
        }
        try await task.send(.string("finalize"))
    }

    public func disconnect() async {
        receiveTask?.cancel()
        receiveTask = nil
        try? await webSocketTask?.send(.string("done"))
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
    }

    /// Verifies the API key against the Cartesia voices endpoint.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.cartesia.ai/voices?limit=1")!)
        request.timeoutInterval = timeout
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue("2026-03-01", forHTTPHeaderField: "Cartesia-Version")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return (false, "No HTTP response received.")
            }
            if (200..<300).contains(http.statusCode) {
                return (true, nil)
            }
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            return (false, message)
        } catch {
            return (false, error.localizedDescription)
        }
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
        case "transcript":
            guard let transcriptText = json["text"] as? String else { return }
            let isFinal = (json["is_final"] as? Bool) ?? false
            if isFinal {
                eventsContinuation?.yield(.committed(text: transcriptText))
            } else {
                eventsContinuation?.yield(.partial(text: transcriptText))
            }

        case "error":
            let message = json["message"] as? String ?? json["title"] as? String ?? "Cartesia streaming error"
            eventsContinuation?.yield(.error(message))

        case "flush_done", "done":
            break

        default:
            break
        }
    }
}
