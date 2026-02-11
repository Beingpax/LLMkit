import Foundation

/// Events emitted by a streaming transcription provider.
public enum StreamingTranscriptionEvent: Sendable {
    /// The streaming session has been established.
    case sessionStarted
    /// A partial (non-final) transcript update.
    case partial(text: String)
    /// A finalized transcript segment.
    case committed(text: String)
    /// An error occurred during streaming.
    case error(String)
}

/// Protocol for streaming transcription providers.
///
/// Each provider manages a WebSocket connection to a cloud transcription service,
/// accepts raw PCM audio chunks, and emits transcription events via an `AsyncStream`.
///
/// Lifecycle: `connect()` → `sendAudioChunk()` (repeated) → `commit()` → `disconnect()`
public protocol StreamingTranscriptionProvider: AnyObject {
    /// Connect to the streaming transcription endpoint.
    ///
    /// - Parameters:
    ///   - apiKey: API key for the provider.
    ///   - model: Model name (e.g. `"scribe_v2_realtime"`, `"nova-3"`).
    ///   - language: Optional language code. Pass `nil` for auto-detect.
    ///   - customVocabulary: Optional custom vocabulary terms for recognition boost.
    func connect(apiKey: String, model: String, language: String?, customVocabulary: [String]) async throws

    /// Send a chunk of raw PCM audio data (16-bit, 16kHz, mono, little-endian).
    func sendAudioChunk(_ data: Data) async throws

    /// Signal the end of audio input to finalize transcription.
    func commit() async throws

    /// Disconnect from the streaming endpoint and clean up resources.
    func disconnect() async

    /// Stream of transcription events from the provider.
    var transcriptionEvents: AsyncStream<StreamingTranscriptionEvent> { get }
}
