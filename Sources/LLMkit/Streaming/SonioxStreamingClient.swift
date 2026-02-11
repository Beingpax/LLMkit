import Foundation

/// Soniox stt-rt-v4 real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://stt-rt.soniox.com/transcribe-websocket`.
/// Sends raw binary PCM audio. API key is sent in the initial config JSON (not HTTP header).
/// Parses token-based responses with `is_final` flags and `<fin>` markers.
public final class SonioxStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var finalText = ""

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
        let urlString = "wss://stt-rt.soniox.com/transcribe-websocket"
        guard let url = URL(string: urlString) else {
            throw LLMKitError.invalidURL(urlString)
        }

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: url)

        self.urlSession = session
        self.webSocketTask = task
        task.resume()

        // Send initial configuration (API key is in the config JSON, not HTTP header)
        try await sendConfiguration(apiKey: apiKey, model: model, language: language, customVocabulary: customVocabulary)

        eventsContinuation?.yield(.sessionStarted)

        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Soniox streaming.")
        }
        // Soniox expects raw binary audio frames
        try await task.send(.data(data))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Soniox streaming.")
        }

        let finalizeMessage: [String: Any] = ["type": "finalize"]
        let jsonData = try JSONSerialization.data(withJSONObject: finalizeMessage)
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
        finalText = ""
    }

    // MARK: - Private

    private func sendConfiguration(apiKey: String, model: String, language: String?, customVocabulary: [String]) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Soniox streaming.")
        }

        var config: [String: Any] = [
            "api_key": apiKey,
            "model": model,
            "audio_format": "pcm_s16le",
            "sample_rate": 16000,
            "num_channels": 1
        ]

        if let language, language != "auto", !language.isEmpty {
            config["language_hints"] = [language]
            config["language_hints_strict"] = true
            config["enable_language_identification"] = true
        } else {
            config["enable_language_identification"] = true
        }

        if !customVocabulary.isEmpty {
            config["context"] = ["terms": customVocabulary]
        }

        let jsonData = try JSONSerialization.data(withJSONObject: config)
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
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }

        // Check for error
        if let errorCode = json["error_code"] as? Int {
            let errorMsg = json["error_message"] as? String ?? "Unknown error (code \(errorCode))"
            eventsContinuation?.yield(.error(errorMsg))
            return
        }

        // Check for finished signal
        if let finished = json["finished"] as? Bool, finished {
            if !finalText.isEmpty {
                eventsContinuation?.yield(.committed(text: finalText))
                finalText = ""
            } else {
                eventsContinuation?.yield(.committed(text: ""))
            }
            return
        }

        // Parse tokens
        guard let tokens = json["tokens"] as? [[String: Any]], !tokens.isEmpty else { return }
        processTokens(tokens)
    }

    private func processTokens(_ tokens: [[String: Any]]) {
        var newFinalText = ""
        var newPartialText = ""
        var sawFinMarker = false

        for token in tokens {
            guard let text = token["text"] as? String else { continue }

            if text == "<fin>" {
                sawFinMarker = true
                continue
            }

            let isFinal = token["is_final"] as? Bool ?? false

            if isFinal {
                newFinalText += text
            } else {
                newPartialText += text
            }
        }

        if !newFinalText.isEmpty {
            finalText += newFinalText
        }

        if sawFinMarker {
            eventsContinuation?.yield(.committed(text: finalText))
            finalText = ""
        } else if !newPartialText.isEmpty {
            let currentPartial = finalText + newPartialText
            eventsContinuation?.yield(.partial(text: currentPartial))
        }
    }
}
