import Foundation

/// Zoom AI Services Scribe Live streaming client speaking the `live-asr` WebSocket subprotocol.
///
/// Connects to `wss://api.zoom.us/v2/aiservices/scribe/live`. Sends `session.update` first and streams raw
/// PCM16 audio only after the server acknowledges with `session.updated`.
///
/// `commit()` ends the session. The full transcript arrives on `finalizationEvents` once the server sends
/// `session.closed`. Finalization has no timeout here, so callers should apply their own.
/// API docs: https://developers.zoom.us/docs/ai-services/scribe/live-mode/
public final class ZoomScribeStreamingClient: StreamingTranscriptionProvider, @unchecked Sendable {

    private static let handshakeTimeout: TimeInterval = 10

    private let trackingIdPrefix: String
    private let textPolishing: Bool

    // Accessed from the receive task as well as the caller, so all mutable state goes through `lock`.
    private let lock = NSLock()
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var receiveTask: Task<Void, Never>?
    private var connectionTrackingId: String?
    /// The model sent in `session.update` for the current connection, used to clean up its text.
    private var connectionApiModel: String?
    private var finalTurns: [String] = []
    private var commitRequested = false
    private var sessionEnded = false
    private var finalizationCompleted = false

    private let eventsContinuation: AsyncStream<StreamingTranscriptionEvent>.Continuation
    private let finalizationContinuation: AsyncStream<String>.Continuation

    public private(set) var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent>
    /// Yields once with the full session transcript after `commit()` and `session.closed`, then finishes.
    /// Finishes without yielding if the session fails first.
    public let finalizationEvents: AsyncStream<String>

    /// Tracking id of the current or most recent connection, for support requests.
    private var trackingId: String? {
        lock.withLock { connectionTrackingId }
    }

    /// - Parameter textPolishing: Opt-in add-on that emits polished text for completed turns. Applies only to en-US
    ///   without language detection. Captions (partials) are never polished.
    public init(trackingIdPrefix: String = "zoom_ai_community_llmkit_", textPolishing: Bool = false) {
        self.trackingIdPrefix = trackingIdPrefix
        self.textPolishing = textPolishing

        let (events, eventsContinuation) = AsyncStream.makeStream(of: StreamingTranscriptionEvent.self)
        transcriptionEvents = events
        self.eventsContinuation = eventsContinuation

        let (finals, finalizationContinuation) = AsyncStream.makeStream(of: String.self)
        finalizationEvents = finals
        self.finalizationContinuation = finalizationContinuation
    }

    deinit {
        receiveTask?.cancel()
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        urlSession?.invalidateAndCancel()
        eventsContinuation.finish()
        finalizationContinuation.finish()
    }

    public func connect(apiKey: String, model: String, language: String?, customVocabulary: [String] = []) async throws {
        try validateAPIKey(apiKey)

        let trackingId = ZoomScribe.makeTrackingId(prefix: trackingIdPrefix)
        let request = ZoomScribe.makeLiveRequest(apiKey: apiKey, trackingId: trackingId)
        let session = URLSession(configuration: .default)
        let socket = session.webSocketTask(with: request)
        let sessionUpdate = ZoomScribe.sessionUpdateMessage(
            language: language, model: model, vocabulary: customVocabulary, textPolishing: textPolishing)

        lock.withLock {
            connectionTrackingId = trackingId
            connectionApiModel = ZoomScribe.apiModel(for: model, language: language)
            webSocketTask = socket
            urlSession = session
            finalTurns = []
            commitRequested = false
            sessionEnded = false
        }
        socket.resume()

        do {
            _ = try await Self.performHandshake(
                socket: socket, sessionUpdate: sessionUpdate, trackingId: trackingId,
                timeout: Self.handshakeTimeout)
        } catch {
            let released = releaseConnection()
            Self.close(released.socket, session: released.session)
            throw error
        }

        eventsContinuation.yield(.sessionStarted)
        startReceiveLoop(socket)
    }

    public func sendAudioChunk(_ data: Data) async throws {
        guard let socket = currentSocket else {
            throw LLMKitError.networkError("Not connected to Zoom Scribe streaming.")
        }
        try await socket.send(.data(data))
    }

    /// Scribe has no flush message. `session.close` makes the server emit any remaining turns and then `session.closed`.
    public func commit() async throws {
        guard let socket = currentSocket else {
            throw LLMKitError.networkError("Not connected to Zoom Scribe streaming.")
        }
        lock.withLock { commitRequested = true }
        try await socket.send(.string(ZoomScribe.sessionCloseMessage))
    }

    public func disconnect() async {
        let released = releaseConnection()
        released.receiveTask?.cancel()
        Self.close(released.socket, session: released.session)
        eventsContinuation.finish()
        finalizationContinuation.finish()
    }

    /// Verifies a Zoom Scribe API key by completing a Live handshake. Sends no model, language, or audio.
    ///
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(
        _ apiKey: String,
        trackingIdPrefix: String = "zoom_ai_community_llmkit_",
        timeout: TimeInterval = 10
    ) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        let trackingId = ZoomScribe.makeTrackingId(prefix: trackingIdPrefix)
        let request = ZoomScribe.makeLiveRequest(apiKey: apiKey, trackingId: trackingId)
        let session = URLSession(configuration: .default)
        let socket = session.webSocketTask(with: request)
        socket.resume()

        do {
            _ = try await performHandshake(
                socket: socket,
                sessionUpdate: ZoomScribe.sessionUpdateMessage(language: nil, model: nil, vocabulary: []),
                trackingId: trackingId,
                timeout: timeout)
        } catch {
            socket.cancel(with: .normalClosure, reason: nil)
            session.invalidateAndCancel()
            return (false, verificationFailureMessage(for: error, trackingId: trackingId))
        }

        try? await socket.send(.string(ZoomScribe.sessionCloseMessage))
        socket.cancel(with: .normalClosure, reason: nil)
        session.invalidateAndCancel()
        return (true, nil)
    }

    /// A failed key check is fixed by the user, not by Zoom support, so the message leaves out the tracking id.
    static func verificationFailureMessage(for error: Error, trackingId: String) -> String {
        if case LLMKitError.httpError(let statusCode, _) = error, statusCode == 401 || statusCode == 403 {
            return "Invalid Zoom Scribe API key"
        }
        return error.localizedDescription.replacingOccurrences(of: " (tracking_id=\(trackingId))", with: "")
    }

    // MARK: - Handshake

    /// Sends `session.update` and waits for `session.updated`. Skips unrelated frames such as `session.created`.
    /// The timeout cancels the socket so the pending receive unblocks.
    private static func performHandshake(
        socket: URLSessionWebSocketTask,
        sessionUpdate: String,
        trackingId: String,
        timeout: TimeInterval
    ) async throws -> String? {
        try await withThrowingTaskGroup(of: String?.self) { group in
            group.addTask {
                do {
                    try await socket.send(.string(sessionUpdate))
                } catch {
                    throw handshakeError(error, socket: socket, trackingId: trackingId)
                }
                while true {
                    let frame = try await Self.receiveFrame(from: socket, trackingId: trackingId)
                    guard let text = Self.text(from: frame),
                          let event = ZoomScribe.parseServerEvent(text)
                    else { continue }
                    switch event {
                    case .sessionUpdated(let model):
                        return model
                    case .error(let code, let message, _):
                        throw LLMKitError.httpError(
                            statusCode: 400,
                            message: ZoomScribe.serverErrorMessage(code: code, message: message, trackingId: trackingId))
                    default:
                        continue
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                socket.cancel(with: .normalClosure, reason: nil)
                throw LLMKitError.timeout
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else {
                throw LLMKitError.timeout
            }
            return result
        }
    }

    private static func receiveFrame(
        from socket: URLSessionWebSocketTask, trackingId: String
    ) async throws -> URLSessionWebSocketTask.Message {
        do {
            return try await socket.receive()
        } catch {
            throw handshakeError(error, socket: socket, trackingId: trackingId)
        }
    }

    /// Maps a send or receive failure during the handshake. Either can be the first to observe a refused upgrade.
    private static func handshakeError(
        _ error: Error, socket: URLSessionWebSocketTask, trackingId: String
    ) -> Error {
        if let http = socket.response as? HTTPURLResponse, http.statusCode != 101 {
            return LLMKitError.httpError(
                statusCode: http.statusCode,
                message: ZoomScribe.withTrackingId("Zoom Scribe refused the connection", trackingId))
        }
        // On Darwin a rejected upgrade (e.g. 401 for an invalid key) often surfaces only as badServerResponse
        // with no HTTP response attached.
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorBadServerResponse {
            return LLMKitError.networkError(ZoomScribe.withTrackingId(
                "Zoom Scribe rejected the connection. Check that the API key is valid", trackingId))
        }
        return LLMKitError.networkError(ZoomScribe.withTrackingId(error.localizedDescription, trackingId))
    }

    private static func text(from message: URLSessionWebSocketTask.Message) -> String? {
        switch message {
        case .string(let text):
            return text
        case .data(let data):
            return String(data: data, encoding: .utf8)
        @unknown default:
            return nil
        }
    }

    // MARK: - Receive loop

    private func startReceiveLoop(_ socket: URLSessionWebSocketTask) {
        let task = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    let frame = try await socket.receive()
                    if let text = Self.text(from: frame) {
                        self?.handle(text)
                    }
                } catch {
                    self?.handleReceiveFailure(error, socket: socket)
                    return
                }
            }
        }
        lock.withLock { receiveTask = task }
    }

    private func handle(_ text: String) {
        guard let event = ZoomScribe.parseServerEvent(text) else { return }

        let apiModel = lock.withLock { connectionApiModel }

        switch event {
        case .delta(let raw):
            let partial = ZoomScribe.cleanedTranscript(raw, model: apiModel)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            // Each delta is the full hypothesis for the in-progress turn, so it replaces the previous partial.
            // It is dropped after commit() because the final text arrives via transcription.completed.
            guard !partial.isEmpty, !lock.withLock({ commitRequested }) else { return }
            eventsContinuation.yield(.partial(text: partial))

        case .transcript(let raw):
            // Turns are joined with a space, so surrounding whitespace is dropped.
            let turn = ZoomScribe.cleanedTranscript(raw, model: apiModel)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !turn.isEmpty else { return }
            lock.withLock { finalTurns.append(turn) }
            eventsContinuation.yield(.committed(text: turn))

        case .error(let code, let message, let fatal):
            eventsContinuation.yield(.error(ZoomScribe.serverErrorMessage(
                code: code, message: message, trackingId: trackingId ?? "")))
            if fatal {
                lock.withLock { sessionEnded = true }
                completeFinalization(with: nil)
            }

        case .sessionClosed:
            let transcript = lock.withLock { () -> String? in
                sessionEnded = true
                return commitRequested ? finalTurns.joined(separator: " ") : nil
            }
            if let transcript {
                completeFinalization(with: transcript)
            }

        case .sessionUpdated, .speechStarted, .other:
            break
        }
    }

    private func handleReceiveFailure(_ error: Error, socket: URLSessionWebSocketTask) {
        // A socket that is no longer current was released by disconnect(), which is not an error.
        let (isCurrent, alreadyEnded) = lock.withLock { () -> (Bool, Bool) in
            guard webSocketTask === socket else { return (false, true) }
            let previous = sessionEnded
            sessionEnded = true
            return (true, previous)
        }
        guard isCurrent else { return }

        if !alreadyEnded {
            eventsContinuation.yield(.error("\(error.localizedDescription) (tracking_id=\(trackingId ?? ""))"))
        }
        completeFinalization(with: nil)
    }

    /// Yields the final transcript at most once, then finishes the finalization stream.
    private func completeFinalization(with transcript: String?) {
        let shouldComplete = lock.withLock { () -> Bool in
            guard !finalizationCompleted else { return false }
            finalizationCompleted = true
            return true
        }
        guard shouldComplete else { return }
        if let transcript {
            finalizationContinuation.yield(transcript)
        }
        finalizationContinuation.finish()
    }

    // MARK: - Private

    private var currentSocket: URLSessionWebSocketTask? {
        lock.withLock { webSocketTask }
    }

    /// Clears connection state under the lock and returns the old handles so the caller can tear them down.
    private func releaseConnection() -> (receiveTask: Task<Void, Never>?, socket: URLSessionWebSocketTask?, session: URLSession?) {
        lock.withLock { () -> (receiveTask: Task<Void, Never>?, socket: URLSessionWebSocketTask?, session: URLSession?) in
            let released = (receiveTask: receiveTask, socket: webSocketTask, session: urlSession)
            receiveTask = nil
            webSocketTask = nil
            urlSession = nil
            finalTurns = []
            commitRequested = false
            sessionEnded = false
            return released
        }
    }

    private static func close(_ socket: URLSessionWebSocketTask?, session: URLSession?) {
        socket?.cancel(with: .normalClosure, reason: nil)
        session?.invalidateAndCancel()
    }
}
