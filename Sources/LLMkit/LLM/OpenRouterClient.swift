import Foundation

/// Client for the OpenRouter API.
///
/// OpenRouter uses the OpenAI-compatible chat completions format, so for chat completions
/// use `OpenAILLMClient` with `https://openrouter.ai/api/v1/chat/completions` as the base URL.
///
/// This client provides OpenRouter-specific functionality like fetching the available model list.
public struct OpenRouterClient: Sendable {

    /// The base URL for OpenRouter's chat completions endpoint.
    public static let chatCompletionsURL = URL(string: "https://openrouter.ai/api/v1/chat/completions")!

    /// Fetches the list of available models from OpenRouter.
    ///
    /// - Parameter timeout: Request timeout in seconds (default 15).
    /// - Returns: A sorted array of model ID strings (e.g. `["anthropic/claude-3-haiku", ...]`).
    public static func fetchModels(timeout: TimeInterval = 15) async throws -> [String] {
        let url = URL(string: "https://openrouter.ai/api/v1/models")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let dataArray = json["data"] as? [[String: Any]] else {
            throw LLMKitError.decodingError("Unexpected response format from OpenRouter models endpoint.")
        }

        let models = dataArray.compactMap { $0["id"] as? String }
        return models.sorted()
    }

    /// Verifies an API key using OpenRouter's key endpoint.
    ///
    /// - Parameters:
    ///   - apiKey: OpenRouter API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(
        _ apiKey: String,
        timeout: TimeInterval = 10
    ) async -> (isValid: Bool, errorMessage: String?) {
        let trimmedKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedKey.isEmpty else {
            return (false, "API key is missing or empty.")
        }

        let url = URL(string: "https://openrouter.ai/api/v1/key")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return (false, "No HTTP response received.")
            }
            if (200..<300).contains(http.statusCode) {
                return (true, nil)
            }
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            return (false, message)
        } catch {
            return (false, error.localizedDescription)
        }
    }
}
