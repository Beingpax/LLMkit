import Foundation

/// Thinking levels supported by Gemini's Interactions API.
public enum GeminiThinkingLevel: String, Encodable, Sendable {
    case minimal
    case low
    case medium
    case high
}

/// Client for Google's native Gemini Interactions API (`/v1/interactions`).
///
/// Uses the stable API, native `x-goog-api-key` authentication, and stateless
/// requests by default. Sampling parameters are intentionally omitted because
/// current Gemini models are optimized for their defaults and newer models
/// deprecate `temperature`, `top_p`, and `top_k`.
public struct GeminiLLMClient: Sendable {
    static let interactionEndpoint = URL(
        string: "https://generativelanguage.googleapis.com/v1/interactions"
    )!
    static let modelsEndpoint = URL(
        string: "https://generativelanguage.googleapis.com/v1/models"
    )!

    /// Sends a text completion request through Google's native Interactions API.
    ///
    /// - Parameters:
    ///   - apiKey: Google AI / Gemini API key.
    ///   - model: Gemini model name, such as `gemini-3.6-flash`.
    ///   - messages: Conversation messages. System messages become the top-level
    ///     `system_instruction`; assistant messages become `model_output` steps.
    ///   - systemPrompt: Explicit system instruction. When present, it takes
    ///     priority over system messages in `messages`.
    ///   - thinkingLevel: Optional native Gemini thinking level.
    ///   - store: Whether Google should store the interaction. Defaults to `false`.
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The final model-output text.
    public static func chatCompletion(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        thinkingLevel: GeminiThinkingLevel? = nil,
        store: Bool = false,
        timeout: TimeInterval = 30
    ) async throws -> String {
        let request = try makeChatCompletionRequest(
            endpoint: interactionEndpoint,
            apiKey: apiKey,
            model: model,
            messages: messages,
            systemPrompt: systemPrompt,
            thinkingLevel: thinkingLevel,
            store: store
        )

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)
        return try decodeChatCompletionResponse(from: data)
    }

    /// Verifies a Gemini API key with the stable native models endpoint.
    public static func verifyAPIKey(
        _ apiKey: String,
        timeout: TimeInterval = 10
    ) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: modelsEndpoint)
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

        do {
            let (data, response) = try await performRequest(
                request,
                timeout: timeout,
                maxRetries: 0
            )
            try validateHTTPResponse(response, data: data)
            return (true, nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }

    static func makeChatCompletionRequest(
        endpoint: URL = interactionEndpoint,
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String?,
        thinkingLevel: GeminiThinkingLevel?,
        store: Bool
    ) throws -> URLRequest {
        try validateAPIKey(apiKey)

        let usableMessages = messages.filter {
            !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let conversationMessages = usableMessages.filter {
            $0.role.lowercased() != "system"
        }

        guard !conversationMessages.isEmpty else {
            throw LLMKitError.encodingError
        }

        let finalRole = conversationMessages.last?.role.lowercased()
        guard finalRole != "assistant", finalRole != "model" else {
            // Gemini does not support prefilled model turns.
            throw LLMKitError.encodingError
        }

        let input = try conversationMessages.map { message in
            let type: String
            switch message.role.lowercased() {
            case "user":
                type = "user_input"
            case "assistant", "model":
                type = "model_output"
            default:
                throw LLMKitError.encodingError
            }

            return GeminiInteractionStep(
                type: type,
                content: [GeminiInteractionContent(type: "text", text: message.content)]
            )
        }

        let explicitSystemPrompt = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let systemInstruction: String?
        if let explicitSystemPrompt, !explicitSystemPrompt.isEmpty {
            systemInstruction = explicitSystemPrompt
        } else {
            let systemMessages = usableMessages
                .filter { $0.role.lowercased() == "system" }
                .map(\.content)
            systemInstruction = systemMessages.isEmpty ? nil : systemMessages.joined(separator: "\n")
        }

        let body = GeminiInteractionRequest(
            model: model,
            input: input,
            systemInstruction: systemInstruction,
            store: store,
            generationConfig: thinkingLevel.map { GeminiGenerationConfig(thinkingLevel: $0) }
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            throw LLMKitError.encodingError
        }
        return request
    }

    static func decodeChatCompletionResponse(from data: Data) throws -> String {
        let response = try decodeJSON(GeminiInteractionResponse.self, from: data)

        for step in response.steps.reversed() where step.type == "model_output" {
            guard let content = step.content else { continue }

            var finalTextBlocks: [String] = []
            for block in content.reversed() {
                if block.type == "text", let text = block.text {
                    finalTextBlocks.insert(text, at: 0)
                } else if !finalTextBlocks.isEmpty {
                    break
                }
            }

            let result = finalTextBlocks.joined()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !result.isEmpty {
                return result
            }
        }

        throw LLMKitError.noResultReturned
    }
}

// MARK: - Request Models

private struct GeminiInteractionRequest: Encodable, Sendable {
    let model: String
    let input: [GeminiInteractionStep]
    let systemInstruction: String?
    let store: Bool
    let generationConfig: GeminiGenerationConfig?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case systemInstruction = "system_instruction"
        case store
        case generationConfig = "generation_config"
    }
}

private struct GeminiInteractionStep: Encodable, Sendable {
    let type: String
    let content: [GeminiInteractionContent]
}

private struct GeminiInteractionContent: Encodable, Sendable {
    let type: String
    let text: String
}

private struct GeminiGenerationConfig: Encodable, Sendable {
    let thinkingLevel: GeminiThinkingLevel

    enum CodingKeys: String, CodingKey {
        case thinkingLevel = "thinking_level"
    }
}

// MARK: - Response Models

private struct GeminiInteractionResponse: Decodable, Sendable {
    let steps: [GeminiResponseStep]
}

private struct GeminiResponseStep: Decodable, Sendable {
    let type: String
    let content: [GeminiResponseContent]?
}

private struct GeminiResponseContent: Decodable, Sendable {
    let type: String
    let text: String?
}
