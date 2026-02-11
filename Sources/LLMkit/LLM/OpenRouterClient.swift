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

    /// Verifies an API key by making a minimal chat completion request to OpenRouter.
    ///
    /// - Parameters:
    ///   - apiKey: OpenRouter API key.
    ///   - model: Model to use for verification (default `"openai/gpt-oss-120b"`).
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(
        _ apiKey: String,
        model: String = "openai/gpt-oss-120b",
        timeout: TimeInterval = 10
    ) async -> (isValid: Bool, errorMessage: String?) {
        await OpenAILLMClient.verifyAPIKey(
            baseURL: chatCompletionsURL,
            apiKey: apiKey,
            model: model,
            timeout: timeout
        )
    }
}
