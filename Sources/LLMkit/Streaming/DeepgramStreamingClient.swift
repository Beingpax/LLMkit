import Foundation

/// Deepgram Nova-3 real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://api.deepgram.com/v1/listen`.
/// Sends raw binary PCM audio (NOT base64). Includes a keepalive timer.
public final class DeepgramStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var keepaliveTask: Task<Void, Never>?
    private var accumulatedFinalText = ""

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>

    public init() {
        var continuation: AsyncStream<StreamingTranscriptionEvent>.Continuation!
        transcriptionEvents = AsyncStream { continuation = $0 }
        eventsContinuation = continuation
    }

    deinit {
        keepaliveTask?.cancel()
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation?.finish()
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        var components = URLComponents(string: "wss://api.deepgram.com/v1/listen")!
        var queryItems: [URLQueryItem] = [
            URLQueryItem(name: "model", value: model),
            URLQueryItem(name: "encoding", value: "linear16"),
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "channels", value: "1"),
            URLQueryItem(name: "smart_format", value: "true"),
            URLQueryItem(name: "numerals", value: "true"),
            URLQueryItem(name: "interim_results", value: "true")
        ]

        if let language, language != "auto", !language.isEmpty {
            queryItems.append(URLQueryItem(name: "language", value: language))
        }

        for term in customVocabulary.prefix(50) {
            queryItems.append(URLQueryItem(name: "keyterm", value: term))
        }

        components.queryItems = queryItems

        guard let url = components.url else {
            throw LLMKitError.invalidURL("wss://api.deepgram.com/v1/listen")
        }

        var request = URLRequest(url: url)
        request.setValue("Token \(apiKey)", forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        eventsContinuation?.yield(.sessionStarted)

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
        keepaliveTask = Task { [weak self] in
            await self?.keepaliveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Deepgram streaming.")
        }
        try await task.send(.data(data))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Deepgram streaming.")
        }

        let finalizeMessage: [String: Any] = ["type": "Finalize"]
        let jsonData = try JSONSerialization.data(withJSONObject: finalizeMessage)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    public func disconnect() async {
        keepaliveTask?.cancel()
        keepaliveTask = nil

        if let task = webSocketTask {
            do {
                let closeMessage: [String: Any] = ["type": "CloseStream"]
                let jsonData = try JSONSerialization.data(withJSONObject: closeMessage)
                let jsonString = String(data: jsonData, encoding: .utf8)!
                try await task.send(.string(jsonString))
            } catch {
                // Ignore errors during disconnect
            }
        }

        receiveTask?.cancel()
        receiveTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        eventsContinuation?.finish()
        accumulatedFinalText = ""
    }

    // MARK: - Private

    private func keepaliveLoop() async {
        do { try await Task.sleep(nanoseconds: 5_000_000_000) } catch { return }

        while !Task.isCancelled {
            guard let task = webSocketTask else { break }
            do {
                let keepaliveMessage: [String: Any] = ["type": "KeepAlive"]
                let jsonData = try JSONSerialization.data(withJSONObject: keepaliveMessage)
                let jsonString = String(data: jsonData, encoding: .utf8)!
                try await task.send(.string(jsonString))
                try await Task.sleep(nanoseconds: 5_000_000_000)
            } catch {
                if !Task.isCancelled { break }
            }
        }
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
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        // Skip control messages
        if let type = json["type"] as? String,
           type == "Metadata" || type == "SpeechStarted" || type == "UtteranceEnd" {
            return
        }

        if let error = json["error"] as? String {
            eventsContinuation?.yield(.error(error))
            return
        }

        guard let channel = json["channel"] as? [String: Any],
              let alternatives = channel["alternatives"] as? [[String: Any]],
              let firstAlternative = alternatives.first,
              let transcript = firstAlternative["transcript"] as? String else { return }

        let isFinal = json["is_final"] as? Bool ?? false
        let speechFinal = json["speech_final"] as? Bool ?? false

        if isFinal || speechFinal {
            if !transcript.isEmpty {
                if !accumulatedFinalText.isEmpty {
                    accumulatedFinalText += " "
                }
                accumulatedFinalText += transcript
                eventsContinuation?.yield(.committed(text: transcript))
            } else {
                eventsContinuation?.yield(.committed(text: ""))
            }
        } else {
            if !transcript.isEmpty {
                let fullPartial = accumulatedFinalText.isEmpty
                    ? transcript
                    : accumulatedFinalText + " " + transcript
                eventsContinuation?.yield(.partial(text: fullPartial))
            }
        }
    }
}
