import Foundation

/// HTTP status codes that warrant automatic retry.
let defaultRetryableStatusCodes: Set<Int> = [429, 500, 502, 503, 504]

// Preserve chat completion retries for all server errors without changing other clients.
let chatRetryableStatusCodes: Set<Int> = Set([429]).union(500...599)

/// Validates that an API key is not empty.
func validateAPIKey(_ apiKey: String) throws {
    guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw LLMKitError.missingAPIKey
    }
}

/// Performs an HTTP request with automatic retry for transient failures.
///
/// Retries on network errors and the selected HTTP status codes with backoff (1s, 2s).
/// When enabled, timeout retries (including HTTP 408) are immediate and share the same attempt budget.
func performRequest(
    _ request: URLRequest,
    timeout: TimeInterval = 30,
    maxRetries: Int = 2,
    retryableStatusCodes: Set<Int> = defaultRetryableStatusCodes,
    retryOnTimeout: Bool = false
) async throws -> (Data, URLResponse) {
    var req = request
    req.timeoutInterval = timeout
    let maxRetries = max(0, maxRetries)
    var lastError: (any Error)?
    var delayBeforeRetry = true

    for attempt in 0...maxRetries {
        try Task.checkCancellation()
        if attempt > 0 && delayBeforeRetry {
            let delay = UInt64(pow(2.0, Double(attempt - 1)) * 1_000_000_000)
            try await Task.sleep(nanoseconds: delay)
        }

        let session = makeEphemeralURLSession(timeout: timeout)
        defer { session.finishTasksAndInvalidate() }

        do {
            let (data, response) = try await session.data(for: req)
            if let http = response as? HTTPURLResponse,
               (retryableStatusCodes.contains(http.statusCode) || (retryOnTimeout && http.statusCode == 408)),
               attempt < maxRetries {
                delayBeforeRetry = http.statusCode != 408
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
            guard retryOnTimeout, attempt < maxRetries else {
                throw LLMKitError.timeout
            }
            lastError = LLMKitError.timeout
            delayBeforeRetry = false
        } catch {
            try Task.checkCancellation()
            let nsError = error as NSError
            if error is CancellationError
                || (nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled)
            {
                throw error
            }
            lastError = error
            delayBeforeRetry = true
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

        let session = makeEphemeralURLSession(timeout: timeout)
        defer { session.finishTasksAndInvalidate() }

        do {
            let (data, response) = try await session.upload(for: req, from: bodyData)
            if let http = response as? HTTPURLResponse,
               defaultRetryableStatusCodes.contains(http.statusCode),
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

private func makeEphemeralURLSession(timeout: TimeInterval) -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.timeoutIntervalForRequest = timeout
    configuration.timeoutIntervalForResource = timeout
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    configuration.urlCache = nil
    return URLSession(configuration: configuration)
}
