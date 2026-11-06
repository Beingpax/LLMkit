import Foundation
import os

private final class SonioxStreamIntegrity: @unchecked Sendable {
    private let lock = NSLock()
    private var interrupted = false

    var requiresFullRecording: Bool { lock.withLock { interrupted } }
    func markInterrupted() { lock.withLock { interrupted = true } }
}

public final class SonioxStreamingClient: StreamingTranscriptionProvider, Sendable {
    public let transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    private let integrity: SonioxStreamIntegrity
    private let connection: SonioxConnection

    /// A restarted or interrupted stream cannot establish complete recording coverage.
    public var requiresFullRecording: Bool { integrity.requiresFullRecording }

    public init(
        endpoint: URL = URL(string: "wss://stt-rt.soniox.com/transcribe-websocket")!,
        sendTimeout: TimeInterval = 5,
        finalizationTimeout: TimeInterval = 3
    ) {
        let events = AsyncStream.makeStream(of: StreamingTranscriptionEvent.self)
        let integrity = SonioxStreamIntegrity()
        self.integrity = integrity
        transcriptionEvents = events.stream
        connection = SonioxConnection(
            endpoint: endpoint, sendTimeout: sendTimeout, finalizationTimeout: finalizationTimeout,
            integrity: integrity, events: events.continuation
        )
    }

    deinit {
        let connection = connection
        Task { await connection.disconnect() }
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        try await connection.connect(key: apiKey, model: model, language: language, vocabulary: customVocabulary)
    }
    public func sendAudioChunk(_ data: Data) async throws { await connection.sendAudio(data) }
    public func commit() async throws { try await connection.commit() }
    /// Sends end-of-audio and waits for Soniox's finished acknowledgement; silence may return empty text.
    public func finalize() async throws -> String { try await connection.finalize() }
    public func disconnect() async { await connection.disconnect() }
}

private actor SonioxConnection {
    private let logger = Logger(subsystem: "LLMkit", category: "SonioxStreamingClient")
    private var vocabulary: [String] = []
    private var model = "stt-rt-v6"
    private let endpoint: URL
    private let finalizationTimeout: TimeInterval
    private let finalResults: AsyncThrowingStream<String, Error>
    private let integrity: SonioxStreamIntegrity
    private let events: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private let final: AsyncThrowingStream<String, Error>.Continuation
    private var session: URLSession?
    private var socket: URLSessionWebSocketTask?
    private var receiver: Task<Void, Never>?
    private var key = ""
    private var language: String?
    private var failure: SonioxStreamingError?
    private var reconnectAttempts = 0
    private var stopped = false
    private var ending = false
    private var text = ""
    private var audio = Data()
    private var sender: Task<Void, Never>?
    private var terminal = false
    private var acceptedBytes = 0
    private var sentBytes = 0
    private var sentFrames = 0
    private let operationTimeoutSeconds: TimeInterval
    private var flushing = false
    private var segments: [String] = []
    // 100 ms of 16-kHz, mono, 16-bit PCM; retain at most 30 seconds during recovery.
    private let frameBytes = 3_200
    private let bufferLimit = 960_000

    init(
        endpoint: URL, sendTimeout: TimeInterval, finalizationTimeout: TimeInterval,
        integrity: SonioxStreamIntegrity,
        events: AsyncStream<StreamingTranscriptionEvent>.Continuation
    ) {
        self.endpoint = endpoint
        self.operationTimeoutSeconds = sendTimeout.isFinite && sendTimeout > 0 ? sendTimeout : 5
        self.finalizationTimeout = finalizationTimeout.isFinite && finalizationTimeout > 0 ? finalizationTimeout : 3
        self.integrity = integrity
        self.events = events
        let results = AsyncThrowingStream.makeStream(of: String.self)
        finalResults = results.stream
        final = results.continuation
    }

    func connect(key: String, model: String, language: String?, vocabulary: [String]) async throws {
        try validateAPIKey(key)
        guard !stopped, socket == nil else { throw SonioxStreamingError.connection("Session is already used.") }
        self.key = key
        self.model = model
        self.language = language
        self.vocabulary = vocabulary
        try await open()
    }

    private func open() async throws {
        try Task.checkCancellation()
        guard !stopped else { throw CancellationError() }
        let session = URLSession(configuration: .ephemeral)
        var request = URLRequest(url: endpoint)
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        let socket = session.webSocketTask(with: request)
        self.session = session
        self.socket = socket
        socket.resume()
        var config: [String: Any] = [
            "model": model, "audio_format": "pcm_s16le", "sample_rate": 16000,
            "num_channels": 1, "enable_language_identification": true
        ]
        if let language, language != "auto", !language.isEmpty {
            config["language_hints"] = [language]
            config["language_hints_strict"] = true
        }
        if !vocabulary.isEmpty { config["context"] = ["terms": vocabulary] }
        do {
            let data = try JSONSerialization.data(withJSONObject: config)
            try await send(.string(String(decoding: data, as: UTF8.self)), to: socket)
            guard !stopped, self.socket === socket else { throw CancellationError() }
            failure = nil
            events.yield(.sessionStarted)
            receiver = Task { [weak self] in
                do {
                    while !Task.isCancelled {
                        let message = try await socket.receive()
                        guard await self?.consume(message, from: socket) == true else { break }
                    }
                } catch {
                    await self?.connectionFailed(error, socket: socket)
                }
            }
        } catch {
            closeSocket()
            throw error
        }
    }

    func sendAudio(_ data: Data) {
        guard !stopped, !ending, !terminal else { return }
        guard audio.count + data.count <= bufferLimit else {
            integrity.markInterrupted()
            terminal = true
            audio.removeAll()
            closeSocket()
            final.finish(throwing: SonioxStreamingError.bufferOverflow)
            events.yield(.error(SonioxStreamingError.bufferOverflow.localizedDescription))
            return
        }
        acceptedBytes += data.count
        audio.append(data)
        startSender()
    }

    private func startSender() {
        guard sender == nil, !terminal, !stopped, !flushing,
            audio.count >= frameBytes || (ending && !audio.isEmpty)
        else { return }
        sender = Task { [weak self] in await self?.sendBufferedAudio() }
    }

    private func sendBufferedAudio() async {
        defer { sender = nil }
        while !stopped, !terminal, !Task.isCancelled {
            guard audio.count >= frameBytes || (ending && !audio.isEmpty) else { return }
            if let failure {
                guard !ending, failure.isRetriable, reconnectAttempts < 3 else {
                    terminal = true
                    return
                }
                reconnectAttempts += 1
                let isConnectionLimit: Bool
                if case .server(let type, _, _) = failure {
                    isConnectionLimit = ["limit_exceeded", "max_concurrent_connections_reached"].contains(type)
                } else { isConnectionLimit = false }
                let delay = isConnectionLimit
                    ? 20.0 : pow(2.0, Double(reconnectAttempts - 1))
                logger.notice("Reconnecting Soniox attempt=\(self.reconnectAttempts)")
                do {
                    try await Task.sleep(for: .seconds(delay))
                    guard !stopped, !ending, !Task.isCancelled else { return }
                    try await open()
                } catch {
                    if error is CancellationError { return }
                    self.failure = (error as? SonioxStreamingError) ?? .connection(error.localizedDescription)
                    continue
                }
            }
            guard let socket else { return }
            let chunk = Data(audio.prefix(frameBytes))
            do {
                try await send(.data(chunk), to: socket)
                guard !stopped, !terminal else { return }
                audio.removeFirst(chunk.count)
                sentBytes += chunk.count
                sentFrames += 1
            } catch {
                connectionFailed(error, socket: socket)
            }
        }
    }

    func commit() async throws {
        try await drainAudio(ending: false)
        guard let socket else { throw SonioxStreamingError.incompleteStream }
        try await send(.string("{\"type\":\"finalize\"}"), to: socket)
    }

    func finalize() async throws -> String {
        do {
            try await drainAudio(ending: true)
            guard let socket else { throw SonioxStreamingError.incompleteStream }
            try await send(.string(""), to: socket)
            let results = finalResults
            let text = try await withStreamingDeadline(
                seconds: finalizationTimeout,
                timeoutError: SonioxStreamingError.timeout(operation: "stream finalization")
            ) {
                var iterator = results.makeAsyncIterator()
                guard let text = try await iterator.next() else { throw CancellationError() }
                return text
            }
            guard !stopped, !integrity.requiresFullRecording else { throw SonioxStreamingError.incompleteStream }
            return text
        } catch {
            integrity.markInterrupted()
            sender?.cancel()
            closeSocket()
            throw error
        }
    }

    private func drainAudio(ending: Bool) async throws {
        if ending { self.ending = true }
        guard !stopped, !terminal, !integrity.requiresFullRecording else {
            throw failure ?? SonioxStreamingError.incompleteStream
        }
        guard !flushing else { throw SonioxStreamingError.connection("A commit is already pending.") }
        startSender()
        flushing = true
        defer {
            flushing = false
            if !self.ending { startSender() }
        }
        do {
            let sender = sender
            try await withStreamingDeadline(
                seconds: operationTimeoutSeconds,
                timeoutError: SonioxStreamingError.timeout(operation: "audio drain")
            ) { await sender?.value }
            guard !stopped, !terminal, !integrity.requiresFullRecording, let socket else {
                throw failure ?? SonioxStreamingError.incompleteStream
            }
            if !audio.isEmpty {
                let tail = audio
                try await send(.data(tail), to: socket)
                guard !stopped, !terminal, !integrity.requiresFullRecording else { throw SonioxStreamingError.incompleteStream }
                audio.removeFirst(tail.count)
                sentBytes += tail.count
                sentFrames += 1
            }
            logger.notice("Soniox audio drained acceptedBytes=\(self.acceptedBytes) sentBytes=\(self.sentBytes) frames=\(self.sentFrames)")
        } catch {
            integrity.markInterrupted()
            self.sender?.cancel()
            closeSocket()
            throw error
        }
    }

    private func send(_ message: URLSessionWebSocketTask.Message, to socket: URLSessionWebSocketTask) async throws {
        try await withStreamingDeadline(
            seconds: operationTimeoutSeconds,
            timeoutError: SonioxStreamingError.timeout(operation: "socket send")
        ) { try await socket.send(message) }
    }

    private func connectionFailed(_ error: Error, socket failedSocket: URLSessionWebSocketTask?) {
        guard !stopped, let failedSocket, socket === failedSocket else { return }
        integrity.markInterrupted()
        failure = (error as? SonioxStreamingError) ?? .connection(error.localizedDescription)
        if !text.isEmpty { events.yield(.committed(text: text)); text = "" }
        events.yield(.error(error.localizedDescription))
        final.finish(throwing: failure ?? SonioxStreamingError.incompleteStream)
        closeSocket()
    }

    private func consume(_ message: URLSessionWebSocketTask.Message, from socket: URLSessionWebSocketTask) -> Bool {
        guard !stopped, self.socket === socket else { return false }
        let data: Data
        switch message {
        case .string(let value): data = Data(value.utf8)
        case .data(let value): data = value
        @unknown default: return true
        }
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return true }
        if let code = json["error_code"] as? Int {
            let type = json["error_type"] as? String
            let requestID = json["request_id"] as? String ?? "unknown"
            logger.error("Soniox error response=\(String(decoding: data, as: UTF8.self), privacy: .private)")
            logger.error("Soniox error code=\(code) type=\(type ?? "unknown", privacy: .public) requestID=\(requestID, privacy: .public)")
            connectionFailed(SonioxStreamingError.server(type: type ?? "unknown", message: json["error_message"] as? String ?? "Soniox request failed", requestID: json["request_id"] as? String), socket: socket)
            return false
        }
        // A valid server response confirms recovery; configuration sends alone do not.
        reconnectAttempts = 0
        var partial = ""
        var sawFinalization = false
        for token in json["tokens"] as? [[String: Any]] ?? [] {
            guard let value = token["text"] as? String else { continue }
            if value == "<fin>" { sawFinalization = true; continue }
            if value == "<end>" { continue }
            if token["is_final"] as? Bool == true { text += value }
            else { partial += value }
        }
        if sawFinalization {
            if !text.isEmpty { segments.append(text) }
            events.yield(.committed(text: text))
            text = ""
        } else if !text.isEmpty || !partial.isEmpty {
            events.yield(.partial(text: text + partial))
        }
        if json["finished"] as? Bool == true {
            guard ending else {
                connectionFailed(SonioxStreamingError.incompleteStream, socket: socket)
                return false
            }
            final.yield((segments + (text.isEmpty ? [] : [text])).joined(separator: " "))
            final.finish()
            return false
        }
        return true
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
        sender?.cancel()
        sender = nil
        audio.removeAll()
        closeSocket()
        key = ""
        events.finish()
        final.finish(throwing: CancellationError())
    }
}
