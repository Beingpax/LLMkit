import Foundation

/// Unified error type for all LLMkit API operations.
public enum LLMKitError: Error, LocalizedError, Sendable {
    /// The API key is missing or empty.
    case missingAPIKey

    /// The constructed URL was invalid.
    case invalidURL(String)

    /// A network-level failure occurred (no HTTP response received).
    case networkError(String)

    /// The server returned a non-2xx HTTP status code.
    case httpError(statusCode: Int, message: String)

    /// The response body could not be decoded into the expected type.
    case decodingError(String)

    /// The API returned a successful response but it contained no usable result.
    case noResultReturned

    /// The request body could not be encoded.
    case encodingError

    /// An async-polling operation exceeded the allowed wait time.
    case timeout

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "API key is missing or empty."
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .networkError(let detail):
            return "Network error: \(detail)"
        case .httpError(let statusCode, let message):
            return "HTTP \(statusCode): \(message)"
        case .decodingError(let detail):
            return "Failed to decode response: \(detail)"
        case .noResultReturned:
            return "The API returned no usable result."
        case .encodingError:
            return "Failed to encode request body."
        case .timeout:
            return "The request timed out."
        }
    }
}
