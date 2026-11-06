import Foundation

/// Scribe V2 streaming with manual, acknowledged segment commits.
public final class ElevenLabsStreamingClient: StreamingTranscriptionProvider, Sendable {
    public let transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    private let connection: ElevenLabsConnection

    /// Deadlines are client safeguards, not ElevenLabs protocol requirements.
    public init(
        endpoint: URL = URL(string: "wss://api.elevenlabs.io/v1/speech-to-text/realtime")!,
        handshakeTimeout: TimeInterval = 10,
        sendTimeout: TimeInterval = 5,
        commitTimeout: TimeInterval = 3
    ) {
        let events = AsyncStream.makeStream(of: StreamingTranscriptionEvent.self)
        transcriptionEvents = events.stream
        connection = ElevenLabsConnection(
            endpoint: endpoint,
            handshakeTimeout: handshakeTimeout,
            sendTimeout: sendTimeout,
            commitTimeout: commitTimeout,
            events: events.continuation
        )
    }

    deinit {
        let connection = connection
        Task { await connection.disconnect() }
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        try await connection.connect(apiKey: apiKey, model: model, language: language, terms: customVocabulary)
    }

    public func sendAudioChunk(_ data: Data) async throws { await connection.enqueue(data) }
    public func commit() async throws { _ = try await connection.flush(ending: false) }

    /// Ends input and returns all acknowledged segments, including a valid empty result for silence.
    public func finalize() async throws -> String { try await connection.flush(ending: true) }
    public func disconnect() async { await connection.disconnect() }
}

private actor ElevenLabsConnection {
    private let endpoint: URL
    private let handshakeTimeout: TimeInterval
    private let sendTimeout: TimeInterval
    private let commitTimeout: TimeInterval
    private let events: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private var session: URLSession?
    private var socket: URLSessionWebSocketTask?
    private var receiver: Task<Void, Never>?
    private var sender: Task<Void, Never>?
    private var failure: ElevenLabsStreamingError?
    private var pendingCommit: AsyncThrowingStream<String, Error>.Continuation?
    private var segments: [String] = []
    private var audio = Data()
    private var segmentBytes = 0
    private var flushing = false
    private var ending = false
    private var stopped = false
    private var ready = false
    private var lastPlainCommit: String?
    private let frameBytes = 3_200 // 100 ms of 16-kHz mono PCM16.
    private let bufferLimit = 960_000 // 30 seconds of pending audio.
    private let segmentLimit = 640_000 // Commit at 20 seconds, before server auto-commit.

    init(
        endpoint: URL, handshakeTimeout: TimeInterval, sendTimeout: TimeInterval,
        commitTimeout: TimeInterval, events: AsyncStream<StreamingTranscriptionEvent>.Continuation
    ) {
        self.endpoint = endpoint
        self.handshakeTimeout = Self.deadline(handshakeTimeout, default: 10)
        self.sendTimeout = Self.deadline(sendTimeout, default: 5)
        self.commitTimeout = Self.deadline(commitTimeout, default: 3)
        self.events = events
    }

    private static func deadline(_ value: TimeInterval, default fallback: TimeInterval) -> TimeInterval {
        value.isFinite && value > 0 ? value : fallback
    }

    func connect(apiKey: String, model: String, language: String?, terms: [String]) async throws {
        try validateAPIKey(apiKey)
        guard !stopped, socket == nil else { throw ElevenLabsStreamingError.connection("Session is already used.") }
        guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw LLMKitError.invalidURL(endpoint.absoluteString)
        }
        var query = [
            URLQueryItem(name: "model_id", value: model),
            URLQueryItem(name: "audio_format", value: "pcm_16000"),
            URLQueryItem(name: "commit_strategy", value: "manual"),
            URLQueryItem(name: "no_verbatim", value: "true")
        ]
        if let language, language != "auto", !language.isEmpty {
            query.append(URLQueryItem(name: "language_code", value: language))
        }
        query += Self.normalizedKeyterms(terms).map { URLQueryItem(name: "keyterms", value: $0) }
        components.queryItems = query
        guard let url = components.url else { throw LLMKitError.invalidURL(endpoint.absoluteString) }
        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")
        let session = URLSession(configuration: .ephemeral)
        let socket = session.webSocketTask(with: request)
        self.session = session
        self.socket = socket
        socket.resume()
        do {
            let message = try await withStreamingDeadline(seconds: handshakeTimeout, timeoutError: ElevenLabsStreamingError.timeout(operation: "handshake")) {
                try await socket.receive()
            }
            let json = try Self.decode(message)
            if let error = Self.serverError(json) { throw error }
            guard json["message_type"] as? String == "session_started" else {
                throw ElevenLabsStreamingError.invalidHandshake
            }
            try Task.checkCancellation()
            guard !stopped, self.socket === socket else { throw CancellationError() }
            ready = true
            events.yield(.sessionStarted)
            receiver = Task { [weak self] in
                do {
                    while !Task.isCancelled {
                        let message = try await socket.receive()
                        guard await self?.consume(message) == true else { break }
                    }
                } catch {
                    await self?.fail(error)
                }
            }
        } catch {
            let reportedError: Error
            if !(error is CancellationError),
               let status = (socket.response as? HTTPURLResponse)?.statusCode,
               [401, 402, 403, 429].contains(status) {
                let type = status == 402 ? "quota_exceeded" : status == 429 ? "rate_limited" : "auth_error"
                reportedError = ElevenLabsStreamingError.server(type: type, message: "WebSocket upgrade rejected (HTTP \(status)).")
            } else {
                reportedError = error
            }
            fail(reportedError)
            events.finish()
            throw reportedError
        }
    }

    func enqueue(_ data: Data) {
        guard ready, !stopped, !ending, failure == nil else { return }
        guard audio.count + data.count <= bufferLimit else { fail(ElevenLabsStreamingError.bufferOverflow); return }
        audio.append(data)
        startSender()
    }

    private func startSender() {
        guard sender == nil, failure == nil, !stopped, !flushing,
              audio.count >= frameBytes || (ending && !audio.isEmpty) else { return }
        sender = Task { [weak self] in await self?.sendBufferedAudio() }
    }

    private func sendBufferedAudio() async {
        defer { sender = nil }
        do {
            while !stopped, failure == nil, !Task.isCancelled,
                  audio.count >= frameBytes || (ending && !audio.isEmpty) {
                let chunk = Data(audio.prefix(frameBytes))
                try await sendAudio(chunk, commit: false)
                try checkState()
                audio.removeFirst(chunk.count)
                segmentBytes += chunk.count
                if segmentBytes >= segmentLimit { try await commitSegment() }
            }
        } catch { fail(error) }
    }

    func flush(ending: Bool) async throws -> String {
        if ending { self.ending = true }
        try checkState()
        guard !flushing else { throw ElevenLabsStreamingError.connection("A commit is already pending.") }
        startSender()
        flushing = true
        defer {
            flushing = false
            if !self.ending { startSender() }
        }
        do {
            let sender = sender
            try await withStreamingDeadline(seconds: sendTimeout + commitTimeout, timeoutError: ElevenLabsStreamingError.timeout(operation: "audio drain")) {
                await sender?.value
            }
            try checkState()
            if !audio.isEmpty {
                let tail = audio
                try await sendAudio(tail, commit: false)
                try checkState()
                audio.removeFirst(tail.count)
                segmentBytes += tail.count
            }
            if segmentBytes > 0 { try await commitSegment() }
            return segments.joined(separator: " ")
        } catch {
            fail(error)
            throw error
        }
    }

    private func commitSegment() async throws {
        guard pendingCommit == nil else { throw ElevenLabsStreamingError.unexpectedCommit }
        let (replies, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        pendingCommit = continuation
        defer { pendingCommit = nil; continuation.finish() }
        try await sendAudio(Data(), commit: true)
        _ = try await withStreamingDeadline(seconds: commitTimeout, timeoutError: ElevenLabsStreamingError.timeout(operation: "commit acknowledgement")) {
            var iterator = replies.makeAsyncIterator()
            guard let text = try await iterator.next() else { throw CancellationError() }
            return text
        }
        try checkState()
        segmentBytes = 0
    }

    private func sendAudio(_ data: Data, commit: Bool) async throws {
        try checkState()
        guard let socket else { throw ElevenLabsStreamingError.connection("Not connected.") }
        let json: [String: Any] = [
            "message_type": "input_audio_chunk", "audio_base_64": data.base64EncodedString(),
            "commit": commit, "sample_rate": 16000
        ]
        let message = String(decoding: try JSONSerialization.data(withJSONObject: json), as: UTF8.self)
        try await withStreamingDeadline(seconds: sendTimeout, timeoutError: ElevenLabsStreamingError.timeout(operation: "audio send")) {
            try await socket.send(.string(message))
        }
    }

    private func consume(_ message: URLSessionWebSocketTask.Message) -> Bool {
        guard !stopped, failure == nil else { return false }
        do {
            let json = try Self.decode(message)
            if let error = Self.serverError(json) { throw error }
            switch json["message_type"] as? String {
            case "partial_transcript":
                lastPlainCommit = nil
                if let text = json["text"] as? String { events.yield(.partial(text: text)) }
            case "committed_transcript", "committed_transcript_with_timestamps":
                guard let text = json["text"] as? String else { throw ElevenLabsStreamingError.unexpectedCommit }
                if json["message_type"] as? String == "committed_transcript_with_timestamps", lastPlainCommit == text { return true }
                guard let pendingCommit else { throw ElevenLabsStreamingError.unexpectedCommit }
                lastPlainCommit = text
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { segments.append(text) }
                events.yield(.committed(text: text))
                self.pendingCommit = nil
                pendingCommit.yield(text)
                pendingCommit.finish()
            default: break
            }
            return true
        } catch { fail(error); return false }
    }

    private func checkState() throws {
        try Task.checkCancellation()
        if let failure { throw failure }
        guard ready, !stopped else { throw CancellationError() }
    }

    private func fail(_ error: Error) {
        guard !stopped, failure == nil else { return }
        let error = (error as? ElevenLabsStreamingError) ?? .connection(error.localizedDescription)
        failure = error
        pendingCommit?.finish(throwing: error)
        events.yield(.error(error.localizedDescription))
        closeSocket()
    }

    private func closeSocket() {
        receiver?.cancel()
        receiver = nil
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        session?.invalidateAndCancel()
        session = nil
    }

    func disconnect() {
        stopped = true
        ready = false
        sender?.cancel()
        sender = nil
        pendingCommit?.finish(throwing: CancellationError())
        pendingCommit = nil
        closeSocket()
        audio.removeAll()
        events.finish()
    }

    private static func decode(_ message: URLSessionWebSocketTask.Message) throws -> [String: Any] {
        let data: Data
        switch message {
        case .string(let text): data = Data(text.utf8)
        case .data(let bytes): data = bytes
        @unknown default: throw ElevenLabsStreamingError.invalidHandshake
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ElevenLabsStreamingError.invalidHandshake
        }
        return json
    }

    private static func serverError(_ json: [String: Any]) -> ElevenLabsStreamingError? {
        guard let type = json["message_type"] as? String,
              ["error", "auth_error", "quota_exceeded", "rate_limited", "resource_exhausted",
               "session_time_limit_exceeded", "input_error", "chunk_size_exceeded", "transcriber_error",
               "commit_throttled", "invalid_request", "unaccepted_terms", "queue_overflow",
               "insufficient_audio_activity"].contains(type) else { return nil }
        return .server(type: type, message: json["message"] as? String ?? type)
    }

    private static func normalizedKeyterms(_ terms: [String]) -> [String] {
        let unsupported = Set("<>{}[]\\")
        var seen = Set<String>()
        return Array(terms.compactMap { term -> String? in
            let value = term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty, value.count <= 20, !value.contains(where: unsupported.contains),
                  seen.insert(value.lowercased()).inserted else { return nil }
            return value
        }.prefix(50))
    }
}
