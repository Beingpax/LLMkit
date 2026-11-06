import Foundation

public enum ElevenLabsStreamingError: Error, LocalizedError, Sendable {
    case invalidHandshake
    case timeout(operation: String)
    case server(type: String, message: String)
    case connection(String)
    case bufferOverflow
    case unexpectedCommit

    public var allowsBatchFallback: Bool {
        guard case .server(let type, _) = self else { return true }
        return !["auth_error", "quota_exceeded", "unaccepted_terms", "invalid_request",
                 "input_error", "chunk_size_exceeded"].contains(type)
    }

    public var errorDescription: String? {
        switch self {
        case .invalidHandshake: return "ElevenLabs did not confirm session_started."
        case .timeout(let operation): return "ElevenLabs timed out during \(operation)."
        case .server(let type, let message): return "ElevenLabs \(type): \(message)"
        case .connection(let message): return "ElevenLabs connection failed: \(message)"
        case .bufferOverflow: return "ElevenLabs could not keep up with the recorded audio."
        case .unexpectedCommit: return "ElevenLabs committed an unexpected segment; stream completeness could not be confirmed."
        }
    }
}
