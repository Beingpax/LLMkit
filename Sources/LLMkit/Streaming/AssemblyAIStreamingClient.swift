import Foundation

/// AssemblyAI Universal Streaming client.
///
/// Connects to `wss://streaming.assemblyai.com/v3/ws` and sends raw PCM16,
/// 16 kHz, mono, little-endian audio frames.
public final class AssemblyAIStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {
    private static let keytermLimit = 100
    private static let minimumChunkBytes = 1_600
    private static let universal3MinTurnSilenceMs = 1_500
    private static let universal3MaxTurnSilenceMs = 4_000
    private static let universalStreamingEndOfTurnConfidenceThreshold = "0.75"
    private static let universalStreamingMinTurnSilenceMs = 2_000
    private static let universalStreamingMaxTurnSilenceMs = 5_000

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation?
    private var receiveTask: Task<Void, Never>?
    private var pendingAudio = Data()
    private var lastCommittedTurnOrder: Int?
    private var didSendTerminate = false

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
        try await connect(apiKey: apiKey, model: model, language: language, prompt: nil, customVocabulary: customVocabulary)
    }

    public func connect(
        apiKey: String,
        model: String,
        language: String?,
        prompt: String?,
        customVocabulary: [String] = []
    ) async throws {
        try validateAPIKey(apiKey)

        guard let url = Self.streamingURL(model: model, language: language, prompt: prompt, customVocabulary: customVocabulary) else {
            throw LLMKitError.invalidURL("wss://streaming.assemblyai.com/v3/ws")
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "Authorization")

        let session = URLSession(configuration: .default)
        let task = session.webSocketTask(with: request)
        urlSession = session
        webSocketTask = task
        pendingAudio.removeAll(keepingCapacity: true)
        lastCommittedTurnOrder = nil
        didSendTerminate = false
        task.resume()

        try await waitForBeginEvent(from: task)

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
        pendingAudio.removeAll(keepingCapacity: false)
        lastCommittedTurnOrder = nil
        didSendTerminate = false
    }

    // MARK: - Private

    private static func streamingURL(model: String, language: String?, prompt: String?, customVocabulary: [String]) -> URL? {
        var components = URLComponents(string: "wss://streaming.assemblyai.com/v3/ws")
        let shouldDetectLanguage = shouldDetectLanguage(language)
        let resolvedStreamingModel = streamingModel(for: model, language: language)
        var queryItems = [
            URLQueryItem(name: "sample_rate", value: "16000"),
            URLQueryItem(name: "encoding", value: "pcm_s16le"),
            URLQueryItem(name: "speech_model", value: resolvedStreamingModel)
        ]

        if isUniversal3Pro(model) {
            queryItems.append(contentsOf: [
                URLQueryItem(name: "min_turn_silence", value: "\(universal3MinTurnSilenceMs)"),
                URLQueryItem(name: "max_turn_silence", value: "\(universal3MaxTurnSilenceMs)"),
                URLQueryItem(name: "vad_threshold", value: "0.4"),
                URLQueryItem(name: "speaker_labels", value: "false"),
                URLQueryItem(name: "language_detection", value: "true"),
                URLQueryItem(name: "u3_rt_pro_vad_threshold", value: "0.5")
            ])
        } else {
            queryItems.append(contentsOf: [
                URLQueryItem(name: "format_turns", value: "true"),
                URLQueryItem(name: "end_of_turn_confidence_threshold", value: universalStreamingEndOfTurnConfidenceThreshold),
                URLQueryItem(name: "min_turn_silence", value: "\(universalStreamingMinTurnSilenceMs)"),
                URLQueryItem(name: "max_turn_silence", value: "\(universalStreamingMaxTurnSilenceMs)")
            ])
            if shouldDetectLanguage && resolvedStreamingModel == "universal-streaming-multilingual" {
                queryItems.append(URLQueryItem(name: "language_detection", value: "true"))
            }
        }

        let supportsPrompt = isUniversal3Pro(model)
        let trimmedPrompt = supportsPrompt ? (prompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "") : ""
        let keyterms = normalizedKeyterms(customVocabulary)
        if !trimmedPrompt.isEmpty {
            queryItems.append(URLQueryItem(name: "prompt", value: trimmedPrompt))
        }
        if supportsKeyterms(model), let keytermsJSON = jsonArrayString(keyterms), !keyterms.isEmpty {
            queryItems.append(URLQueryItem(name: "keyterms_prompt", value: keytermsJSON))
        }

        components?.queryItems = queryItems
        return components?.url
    }

    private static func isUniversal3Pro(_ model: String) -> Bool {
        model == "universal-3-pro" || model == "u3-rt-pro"
    }

    private static func shouldDetectLanguage(_ language: String?) -> Bool {
        guard let language else { return true }
        return language.isEmpty || language == "auto"
    }

    private static func supportsKeyterms(_ model: String) -> Bool {
        switch model {
        case "universal-3-pro", "u3-rt-pro", "universal-streaming", "universal-streaming-english", "universal-streaming-multilingual":
            return true
        default:
            return false
        }
    }

    private static func streamingModel(for model: String, language: String?) -> String {
        if model == "universal-3-pro" || model == "u3-rt-pro" {
            return "u3-rt-pro"
        }
        if model == "universal-streaming-english" || model == "universal-streaming-multilingual" || model == "whisper-rt" {
            return model
        }
        guard let language, !language.isEmpty, language != "auto" else {
            return "universal-streaming-multilingual"
        }
        return language == "en" ? "universal-streaming-english" : "universal-streaming-multilingual"
    }

    private static func normalizedKeyterms(_ customVocabulary: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for term in customVocabulary {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed.count <= 50 else { continue }
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

                if let error = json["error"] as? String {
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

        while !Task.isCancelled {
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

    private func handleMessage(_ json: [String: Any]) {
        if let error = json["error"] as? String {
            eventsContinuation?.yield(.error(error))
            return
        }

        guard let type = json["type"] as? String else { return }
        switch type {
        case "Turn":
            handleTurn(json)
        case "Termination":
            eventsContinuation?.yield(.committed(text: ""))
        default:
            break
        }
    }

    private func handleTurn(_ json: [String: Any]) {
        let transcript = (json["transcript"] as? String) ?? ""
        guard !transcript.isEmpty else { return }

        let endOfTurn = (json["end_of_turn"] as? Bool) ?? false
        let turnIsFormatted = (json["turn_is_formatted"] as? Bool) ?? false
        let turnOrder = json["turn_order"] as? Int

        if endOfTurn && (turnIsFormatted || lastCommittedTurnOrder != turnOrder) {
            eventsContinuation?.yield(.committed(text: transcript))
            lastCommittedTurnOrder = turnOrder
        } else if !endOfTurn {
            eventsContinuation?.yield(.partial(text: transcript))
        }
    }
}
