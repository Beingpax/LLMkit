import Foundation

/// HTTP status codes that warrant automatic retry.
private let retryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]

/// Validates that an API key is not empty.
func validateAPIKey(_ apiKey: String) throws {
    guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw LLMKitError.missingAPIKey
    }
}

/// Performs an HTTP request with automatic retry for transient failures.
///
/// Retries on network errors and HTTP 429/5xx responses with exponential backoff (1s, 2s).
func performRequest(
    _ request: URLRequest,
    timeout: TimeInterval = 30,
    maxRetries: Int = 2
) async throws -> (Data, URLResponse) {
    var req = request
    req.timeoutInterval = timeout
    var lastError: (any Error)?

    for attempt in 0...maxRetries {
        if attempt > 0 {
            let delay = UInt64(pow(2.0, Double(attempt - 1)) * 1_000_000_000)
            try await Task.sleep(nanoseconds: delay)
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            if let http = response as? HTTPURLResponse,
               retryableStatusCodes.contains(http.statusCode),
               attempt < maxRetries {
                lastError = LLMKitError.httpError(
                    statusCode: http.statusCode,
                    message: String(data: data, encoding: .utf8) ?? ""
                )
                continue
            }
            return (data, response)
        } catch let error as LLMKitError {
            throw error
        } catch let error as NSError where error.domain == NSURLErrorDomain && error.code == NSURLErrorTimedOut {
            throw LLMKitError.timeout
        } catch {
            lastError = error
            if attempt < maxRetries { continue }
        }
    }

    throw lastError ?? LLMKitError.networkError("Request failed after \(maxRetries + 1) attempts")
}

/// Performs an HTTP upload with automatic retry for transient failures.
///
/// Retries on network errors and HTTP 429/5xx responses with exponential backoff (1s, 2s).
func performUpload(
    _ request: URLRequest,
    data bodyData: Data,
    timeout: TimeInterval = 30,
    maxRetries: Int = 2
) async throws -> (Data, URLResponse) {
    var req = request
    req.timeoutInterval = timeout
    var lastError: (any Error)?

    for attempt in 0...maxRetries {
        if attempt > 0 {
            let delay = UInt64(pow(2.0, Double(attempt - 1)) * 1_000_000_000)
            try await Task.sleep(nanoseconds: delay)
        }

        do {
            let (data, response) = try await URLSession.shared.upload(for: req, from: bodyData)
            if let http = response as? HTTPURLResponse,
               retryableStatusCodes.contains(http.statusCode),
               attempt < maxRetries {
                lastError = LLMKitError.httpError(
                    statusCode: http.statusCode,
                    message: String(data: data, encoding: .utf8) ?? ""
                )
                continue
            }
            return (data, response)
        } catch let error as LLMKitError {
            throw error
        } catch let error as NSError where error.domain == NSURLErrorDomain && error.code == NSURLErrorTimedOut {
            throw LLMKitError.timeout
        } catch {
            lastError = error
            if attempt < maxRetries { continue }
        }
    }

    throw lastError ?? LLMKitError.networkError("Request failed after \(maxRetries + 1) attempts")
}
