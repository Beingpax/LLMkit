import Foundation

/// Cartesia Ink 2 real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.cartesia.ai/stt/turns/websocket`.
/// Sends raw binary PCM audio (16-bit signed little-endian, 16 kHz).
/// Uses Cartesia turn events for partial and final transcript delivery.
/// API docs: https://docs.cartesia.ai/api-reference/stt/turns/websocket
public final class CartesiaStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {
    public static let defaultEndpoint = "wss://api.cartesia.ai/stt/turns/websocket"
    public static let defaultCartesiaVersion = "2026-03-01"

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var didSendClose = false
    private let endpoint: String
    private let cartesiaVersion: String

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    public init(
        endpoint: String = CartesiaStreamingClient.defaultEndpoint,
        cartesiaVersion: String = CartesiaStreamingClient.defaultCartesiaVersion
    ) {
        self.endpoint = endpoint
        self.cartesiaVersion = cartesiaVersion

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

    /// Ink 2 auto-turn STT is English-only and does not support custom vocabulary.
    /// Those parameters are accepted for protocol conformance but ignored.
    public func connect(apiKey: String, model: String, language _: String?, customVocabulary _: [String] = []) async throws {
        guard var components = URLComponents(string: endpoint) else {
            throw LLMKitError.invalidURL(endpoint)
        }

        let queryItems = [
            URLQueryItem(name: "model", value: model),
            URLQueryItem(name: "encoding", value: "pcm_s16le"),
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "cartesia_version", value: cartesiaVersion),
        ]

        components.queryItems = queryItems

        guard let url = components.url else {
            throw LLMKitError.invalidURL(endpoint)
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)
        self.urlSession = session
        self.webSocketTask = task
        didSendClose = false
        task.resume()

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

    /// Closes the Ink 2 stream so Cartesia processes any buffered audio into turn events.
    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Cartesia streaming.")
        }
        try await sendCloseCommand(on: task)
    }

    public func disconnect() async {
        if let task = webSocketTask, !didSendClose {
            try? await sendCloseCommand(on: task)
        }

        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        didSendClose = false
    }

    /// Verifies the API key against the Cartesia voices endpoint.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.cartesia.ai/voices?limit=1")!)
        request.timeoutInterval = timeout
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue(Self.defaultCartesiaVersion, forHTTPHeaderField: "Cartesia-Version")

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

    private func sendCloseCommand(on task: URLSessionWebSocketTask) async throws {
        guard !didSendClose else { return }
        try await task.send(.string(#"{"type":"close"}"#))
        didSendClose = true
    }

    private func receiveLoop() async {
        guard let task = webSocketTask else { return }
        defer {
            eventsContinuation?.finish()
        }

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
        case "connected":
            eventsContinuation?.yield(.sessionStarted)

        case "turn.update", "turn.eager_end":
            guard let transcriptText = json["transcript"] as? String else { return }
            eventsContinuation?.yield(.partial(text: transcriptText))

        case "turn.end":
            guard let transcriptText = json["transcript"] as? String else { return }
            eventsContinuation?.yield(.committed(text: transcriptText))

        case "error":
            let message = json["message"] as? String ?? json["title"] as? String ?? "Cartesia streaming error"
            eventsContinuation?.yield(.error(message))

        case "turn.start", "turn.resume":
            break

        default:
            break
        }
    }
}
