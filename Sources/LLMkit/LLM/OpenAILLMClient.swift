import Foundation

/// Client for OpenAI-compatible chat completions APIs (`/v1/chat/completions`).
///
/// Works with any provider that implements the OpenAI chat completions format, including:
/// Groq, Cerebras, Gemini (via OpenAI proxy), OpenAI, Mistral, OpenRouter, Ollama, and custom endpoints.
public struct OpenAILLMClient: Sendable {

    /// Sends a chat completion request to an OpenAI-compatible API.
    ///
    /// - Parameters:
    ///   - baseURL: Provider's chat completions URL (e.g. `https://api.groq.com/openai/v1/chat/completions`).
    ///   - apiKey: API key for the provider.
    ///   - model: Model name (e.g. `"llama-3.3-70b-versatile"`).
    ///   - messages: Array of chat messages.
    ///   - systemPrompt: Optional system prompt. If provided, prepended as a system message.
    ///   - temperature: Sampling temperature (default 0.3).
    ///   - reasoningEffort: Optional reasoning effort parameter for supported models.
    ///   - extraBody: Optional dictionary of additional parameters to include in the request body (e.g. `["disable_reasoning": true]`).
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The assistant's response text.
    public static func chatCompletion(
        baseURL: URL,
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        temperature: Double = 0.3,
        reasoningEffort: String? = nil,
        extraBody: [String: Any]? = nil,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        // Prepend system prompt if provided explicitly
        var allMessages = messages
        if let systemPrompt, !systemPrompt.isEmpty {
            allMessages.insert(.system(systemPrompt), at: 0)
        }

        var bodyDict: [String: Any] = [
            "model": model,
            "messages": allMessages.map { ["role": $0.role, "content": $0.content] },
            "temperature": temperature,
            "stream": false
        ]

        if let reasoningEffort {
            bodyDict["reasoning_effort"] = reasoningEffort
        }

        if let extraBody {
            for (key, value) in extraBody {
                bodyDict[key] = value
            }
        }

        guard let body = try? JSONSerialization.data(withJSONObject: bodyDict) else {
            throw LLMKitError.encodingError
        }
        request.httpBody = body

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(OpenAIChatResponse.self, from: data)
        return decoded.choices.first?.message.content ?? ""
    }

    /// Verifies an API key against an OpenAI-compatible provider.
    ///
    /// Makes a minimal chat completion request to confirm the key works.
    ///
    /// - Parameters:
    ///   - baseURL: Provider's chat completions URL.
    ///   - apiKey: API key to verify.
    ///   - model: Model to use for the test request.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(
        baseURL: URL,
        apiKey: String,
        model: String,
        timeout: TimeInterval = 10
    ) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let body: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": "test"]]
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return (false, "No HTTP response received.")
            }
            if http.statusCode == 200 {
                return (true, nil)
            }
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            return (false, message)
        } catch {
            return (false, error.localizedDescription)
        }
    }
}

// MARK: - Response Models

private struct OpenAIChatResponse: Decodable, Sendable {
    let choices: [OpenAIChatChoice]
}

private struct OpenAIChatChoice: Decodable, Sendable {
    let message: OpenAIChatMessage
}

private struct OpenAIChatMessage: Decodable, Sendable {
    let content: String
}
