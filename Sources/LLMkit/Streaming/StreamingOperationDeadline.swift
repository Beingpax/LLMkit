import Foundation

/// An unresponsive socket must not hold up timeout or cancellation cleanup.
func withStreamingDeadline<T: Sendable>(
    seconds: TimeInterval, timeoutError: any Error,
    body: @escaping @Sendable () async throws -> T
) async throws -> T {
    let (results, continuation) = AsyncThrowingStream<T, Error>.makeStream()
    let worker = Task {
        do { continuation.yield(try await body()); continuation.finish() }
        catch { continuation.finish(throwing: error) }
    }
    let timer = Task {
        do {
            try await Task.sleep(for: .seconds(seconds))
            continuation.finish(throwing: timeoutError)
        } catch {}
    }
    defer { worker.cancel(); timer.cancel(); continuation.finish() }
    return try await withTaskCancellationHandler {
        var iterator = results.makeAsyncIterator()
        guard let result = try await iterator.next() else { throw CancellationError() }
        return result
    } onCancel: {
        worker.cancel()
        timer.cancel()
        continuation.finish(throwing: CancellationError())
    }
}
