import Foundation

public enum SonioxStreamingError: Error, LocalizedError, Sendable {
    case server(type: String, message: String, requestID: String?)
    case connection(String)
    case timeout(operation: String)
    case bufferOverflow
    case incompleteStream

    public var isRetriable: Bool {
        switch self {
        case .connection, .timeout: return true
        case .server(let type, _, _):
            return ["service_unavailable", "internal_error", "request_timeout", "max_duration_reached",
                    "limit_exceeded", "max_concurrent_connections_reached"].contains(type)
        case .bufferOverflow, .incompleteStream: return false
        }
    }

    public var errorDescription: String? {
        switch self {
        case .server(let type, let message, _): return "Soniox \(type): \(message)"
        case .connection(let message): return "Soniox connection failed: \(message)"
        case .timeout(let operation): return "Soniox timed out during \(operation)."
        case .bufferOverflow: return "Soniox audio exceeded its recovery buffer limit."
        case .incompleteStream: return "Soniox could not confirm a complete streaming transcript."
        }
    }
}
