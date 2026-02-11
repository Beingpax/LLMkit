import Foundation

/// Client for the Anthropic Messages API (`/v1/messages`).
///
/// Handles Claude models with Anthropic's unique request format (system as top-level field,
/// `x-api-key` header, `anthropic-version` header).
public struct AnthropicLLMClient: Sendable {

    /// Sends a chat completion request to the Anthropic API.
    ///
    /// - Parameters:
    ///   - apiKey: Anthropic API key.
    ///   - model: Model name (e.g. `"claude-sonnet-4-5"`, `"claude-haiku-4-5"`).
    ///   - messages: Array of chat messages. System messages are extracted and sent as the top-level `system` field.
    ///   - systemPrompt: Optional explicit system prompt. If provided, takes priority over system messages in the array.
    ///   - maxTokens: Maximum tokens in the response (default 8192).
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The assistant's response text.
    public static func chatCompletion(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        maxTokens: Int = 8192,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let url = URL(string: "https://api.anthropic.com/v1/messages")!

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        // Extract system prompt: explicit parameter takes priority, then system messages from array
        let system: String?
        let nonSystemMessages: [ChatMessage]
        if let systemPrompt {
            system = systemPrompt
            nonSystemMessages = messages.filter { $0.role != "system" }
        } else {
            let systemMessages = messages.filter { $0.role == "system" }
            system = systemMessages.isEmpty ? nil : systemMessages.map(\.content).joined(separator: "\n")
            nonSystemMessages = messages.filter { $0.role != "system" }
        }

        let body = AnthropicRequest(
            model: model,
            max_tokens: maxTokens,
            system: system,
            messages: nonSystemMessages.map { AnthropicMessage(role: $0.role, content: $0.content) }
        )

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            throw LLMKitError.encodingError
        }

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(AnthropicResponse.self, from: data)
        return decoded.content.first { $0.type == "text" }?.text ?? ""
    }

    /// Verifies that an Anthropic API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: Anthropic API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        let url = URL(string: "https://api.anthropic.com/v1/messages")!

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

        let body = AnthropicRequest(
            model: "claude-haiku-4-5",
            max_tokens: 1,
            system: nil,
            messages: [AnthropicMessage(role: "user", content: "Hi")]
        )
        request.httpBody = try? JSONEncoder().encode(body)

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

// MARK: - Request/Response Models

private struct AnthropicMessage: Codable, Sendable {
    let role: String
    let content: String
}

private struct AnthropicRequest: Encodable, Sendable {
    let model: String
    let max_tokens: Int
    let system: String?
    let messages: [AnthropicMessage]

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encode(max_tokens, forKey: .max_tokens)
        if let system { try container.encode(system, forKey: .system) }
        try container.encode(messages, forKey: .messages)
    }

    private enum CodingKeys: String, CodingKey {
        case model, max_tokens, system, messages
    }
}

private struct AnthropicContentBlock: Decodable, Sendable {
    let type: String
    let text: String?
}

private struct AnthropicResponse: Decodable, Sendable {
    let content: [AnthropicContentBlock]
}
