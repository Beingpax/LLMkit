import Foundation

/// Speechmatics real-time streaming transcription client.
///
/// Connects via WebSocket to `wss://eu2.rt.speechmatics.com/v2`.
/// Sends raw binary PCM audio (NOT base64). Uses `StartRecognition` / `EndOfStream` protocol.
/// API docs: https://docs.speechmatics.com/api-ref/realtime-transcription-websocket
public final class SpeechmaticsStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var accumulatedFinalText = ""
    /// Client-side counter tracking the number of audio chunks sent.
    /// Used as `last_seq_no` in `EndOfStream` (matches the Python SDK behavior).
    private var seqNo = 0

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
        let urlString = "wss://eu2.rt.speechmatics.com/v2"
        guard let url = URL(string: urlString) else {
            throw LLMKitError.invalidURL(urlString)
        }

        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)

        self.urlSession = session
        self.webSocketTask = task
        self.seqNo = 0
        self.accumulatedFinalText = ""
        task.resume()

        // Send StartRecognition and wait for RecognitionStarted BEFORE starting the receive loop.
        try await sendStartRecognition(language: language, operatingPoint: model, customVocabulary: customVocabulary)
        try await waitForRecognitionStarted()

        // Now start the background receive loop for transcription events
        receiveTask = Task { [weak self] in
            await self?.receiveLoop()
        }

        eventsContinuation?.yield(.sessionStarted)
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Speechmatics streaming.")
        }
        seqNo += 1
        try await task.send(.data(data))
    }

    public func commit() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Speechmatics streaming.")
        }

        let endMessage: [String: Any] = [
            "message": "EndOfStream",
            "last_seq_no": seqNo
        ]
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
        accumulatedFinalText = ""
        seqNo = 0
    }

    // MARK: - Private

    private func sendStartRecognition(language: String?, operatingPoint: String, customVocabulary: [String]) async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Speechmatics streaming.")
        }

        let lang = mapLanguage(language)

        var transcriptionConfig: [String: Any] = [
            "language": lang,
            "enable_partials": true,
            "operating_point": operatingPoint,
            "max_delay": 2.0,
            "max_delay_mode": "flexible"
        ]

        if !customVocabulary.isEmpty {
            transcriptionConfig["additional_vocab"] = customVocabulary.map { ["content": $0] }
        }

        let startMessage: [String: Any] = [
            "message": "StartRecognition",
            "audio_format": [
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 16000
            ],
            "transcription_config": transcriptionConfig
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: startMessage)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        try await task.send(.string(jsonString))
    }

    /// Waits for RecognitionStarted from the server. Called before receiveLoop starts,
    /// so there is no race condition on message consumption.
    private func waitForRecognitionStarted() async throws {
        guard let task = webSocketTask else {
            throw LLMKitError.networkError("Not connected to Speechmatics streaming.")
        }

        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            do {
                let message = try await task.receive()
                switch message {
                case .string(let text):
                    guard let data = text.data(using: .utf8),
                          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let messageType = json["message"] as? String else { continue }

                    if messageType == "RecognitionStarted" {
                        return
                    } else if messageType == "Error" {
                        let reason = json["reason"] as? String ?? "Unknown error"
                        throw LLMKitError.httpError(statusCode: 400, message: "Speechmatics: \(reason)")
                    }
                case .data:
                    continue
                @unknown default:
                    continue
                }
            } catch let error as LLMKitError {
                throw error
            } catch {
                throw LLMKitError.networkError("Failed to receive RecognitionStarted: \(error.localizedDescription)")
            }
        }
        throw LLMKitError.timeout
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
              let messageType = json["message"] as? String else {
            return
        }

        switch messageType {
        case "AddPartialTranscript":
            guard let metadata = json["metadata"] as? [String: Any],
                  let transcript = metadata["transcript"] as? String,
                  !transcript.trimmingCharacters(in: .whitespaces).isEmpty else { return }
            let fullPartial = accumulatedFinalText + transcript
            eventsContinuation?.yield(.partial(text: cleanPunctuation(fullPartial.trimmingCharacters(in: .whitespaces))))

        case "AddTranscript":
            guard let metadata = json["metadata"] as? [String: Any],
                  let transcript = metadata["transcript"] as? String else { return }
            accumulatedFinalText += transcript

        case "EndOfTranscript":
            let finalText = cleanPunctuation(accumulatedFinalText.trimmingCharacters(in: .whitespaces))
            eventsContinuation?.yield(.committed(text: finalText))
            accumulatedFinalText = ""

        case "Error":
            let reason = json["reason"] as? String ?? "Unknown error"
            eventsContinuation?.yield(.error(reason))

        default:
            break
        }
    }

    /// Removes errant spaces before punctuation marks that Speechmatics
    /// `metadata.transcript` sometimes includes (e.g. "Hello ." → "Hello.").
    private func cleanPunctuation(_ text: String) -> String {
        text.replacingOccurrences(of: " \\s*([.,!?;:'])", with: "$1", options: .regularExpression)
    }

    /// Maps VoiceInk language codes to Speechmatics language codes.
    /// Speechmatics real-time API does not support "auto" — defaults to "en".
    private func mapLanguage(_ language: String?) -> String {
        guard let language, !language.isEmpty, language != "auto" else { return "en" }
        switch language {
        case "zh": return "cmn"
        default: return language
        }
    }
}
