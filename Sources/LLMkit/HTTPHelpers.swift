import Foundation

/// Validates that an HTTP response has a 2xx status code, otherwise throws `LLMKitError.httpError`.
func validateHTTPResponse(_ response: URLResponse, data: Data) throws {
    guard let http = response as? HTTPURLResponse else {
        throw LLMKitError.networkError("No HTTP response received.")
    }
    guard (200..<300).contains(http.statusCode) else {
        let message = String(data: data, encoding: .utf8) ?? "No error details"
        throw LLMKitError.httpError(statusCode: http.statusCode, message: message)
    }
}

/// Decodes JSON data into the specified `Decodable` type, wrapping decode failures in `LLMKitError.decodingError`.
func decodeJSON<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    do {
        return try JSONDecoder().decode(type, from: data)
    } catch {
        throw LLMKitError.decodingError(error.localizedDescription)
    }
}
