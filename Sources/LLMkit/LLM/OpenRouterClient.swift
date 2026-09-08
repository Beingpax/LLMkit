import Foundation

public struct OpenRouterReasoningCapabilities: Codable, Sendable, Hashable {
    public let mandatory: Bool
    public let defaultEnabled: Bool
    public let supportedEfforts: [String]
    public let defaultEffort: String?
    public let supportsMaxTokens: Bool

    enum CodingKeys: String, CodingKey {
        case mandatory
        case defaultEnabled = "default_enabled"
        case supportedEfforts = "supported_efforts"
        case defaultEffort = "default_effort"
        case supportsMaxTokens = "supports_max_tokens"
    }

    public init(
        mandatory: Bool = false,
        defaultEnabled: Bool = false,
        supportedEfforts: [String] = [],
        defaultEffort: String? = nil,
        supportsMaxTokens: Bool = false
    ) {
        self.mandatory = mandatory
        self.defaultEnabled = defaultEnabled
        self.supportedEfforts = supportedEfforts
        self.defaultEffort = defaultEffort
        self.supportsMaxTokens = supportsMaxTokens
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        mandatory = try container.decodeIfPresent(Bool.self, forKey: .mandatory) ?? false
        defaultEnabled = try container.decodeIfPresent(Bool.self, forKey: .defaultEnabled) ?? false
        supportedEfforts = try container.decodeIfPresent([String].self, forKey: .supportedEfforts) ?? []
        defaultEffort = try container.decodeIfPresent(String.self, forKey: .defaultEffort)
        supportsMaxTokens = try container.decodeIfPresent(Bool.self, forKey: .supportsMaxTokens) ?? false
    }
}

public struct OpenRouterModel: Codable, Sendable, Hashable {
    public let id: String
    public let name: String?
    public let supportedParameters: [String]
    public let reasoning: OpenRouterReasoningCapabilities?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case supportedParameters = "supported_parameters"
        case reasoning
    }

    public init(
        id: String,
        name: String? = nil,
        supportedParameters: [String] = [],
        reasoning: OpenRouterReasoningCapabilities? = nil
    ) {
        self.id = id
        self.name = name
        self.supportedParameters = supportedParameters
        self.reasoning = reasoning
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        supportedParameters = try container.decodeIfPresent([String].self, forKey: .supportedParameters) ?? []
        reasoning = try container.decodeIfPresent(OpenRouterReasoningCapabilities.self, forKey: .reasoning)
    }
}

public struct OpenRouterReasoning: Codable, Sendable, Hashable {
    public let effort: String?
    public let maxTokens: Int?
    public let enabled: Bool?
    public let exclude: Bool?

    public init(
        effort: String? = nil,
        maxTokens: Int? = nil,
        enabled: Bool? = nil,
        exclude: Bool? = nil
    ) {
        self.effort = effort
        self.maxTokens = maxTokens
        self.enabled = enabled
        self.exclude = exclude
    }

    enum CodingKeys: String, CodingKey {
        case effort
        case maxTokens = "max_tokens"
        case enabled
        case exclude
    }
}

public struct OpenRouterPercentileThresholds: Codable, Sendable, Hashable {
    public let p50: Double?
    public let p75: Double?
    public let p90: Double?
    public let p99: Double?

    public init(p50: Double? = nil, p75: Double? = nil, p90: Double? = nil, p99: Double? = nil) {
        self.p50 = p50
        self.p75 = p75
        self.p90 = p90
        self.p99 = p99
    }
}

public struct OpenRouterProviderPreferences: Codable, Sendable, Hashable {
    public enum Sort: String, Codable, Sendable, Hashable {
        case price
        case throughput
        case latency
    }

    public let sort: Sort?
    public let preferredMaxLatency: OpenRouterPercentileThresholds?
    public let preferredMinThroughput: OpenRouterPercentileThresholds?
    public let requireParameters: Bool?
    public let allowFallbacks: Bool?

    public init(
        sort: Sort? = nil,
        preferredMaxLatency: OpenRouterPercentileThresholds? = nil,
        preferredMinThroughput: OpenRouterPercentileThresholds? = nil,
        requireParameters: Bool? = nil,
        allowFallbacks: Bool? = nil
    ) {
        self.sort = sort
        self.preferredMaxLatency = preferredMaxLatency
        self.preferredMinThroughput = preferredMinThroughput
        self.requireParameters = requireParameters
        self.allowFallbacks = allowFallbacks
    }

    enum CodingKeys: String, CodingKey {
        case sort
        case preferredMaxLatency = "preferred_max_latency"
        case preferredMinThroughput = "preferred_min_throughput"
        case requireParameters = "require_parameters"
        case allowFallbacks = "allow_fallbacks"
    }
}

public struct OpenRouterUsage: Decodable, Sendable, Hashable {
    public let promptTokens: Int?
    public let completionTokens: Int?
    public let reasoningTokens: Int?

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case completionTokensDetails = "completion_tokens_details"
    }

    private enum DetailsCodingKeys: String, CodingKey {
        case reasoningTokens = "reasoning_tokens"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        promptTokens = try container.decodeIfPresent(Int.self, forKey: .promptTokens)
        completionTokens = try container.decodeIfPresent(Int.self, forKey: .completionTokens)
        if let details = try? container.nestedContainer(
            keyedBy: DetailsCodingKeys.self,
            forKey: .completionTokensDetails
        ) {
            reasoningTokens = try details.decodeIfPresent(Int.self, forKey: .reasoningTokens)
        } else {
            reasoningTokens = nil
        }
    }
}

public struct OpenRouterMetadata: Decodable, Sendable, Hashable {
    public let requested: String?
    public let strategy: String?
    public let region: String?
    public let summary: String?
    public let attempt: Int?
    public let isBYOK: Bool?
    public let attempts: [Attempt]?

    public struct Attempt: Decodable, Sendable, Hashable {
        public let provider: String?
        public let model: String?
        public let status: Int?
    }

    enum CodingKeys: String, CodingKey {
        case requested
        case strategy
        case region
        case summary
        case attempt
        case isBYOK = "is_byok"
        case attempts
    }
}

public struct OpenRouterCompletion: Sendable, Hashable {
    public let text: String
    public let model: String?
    public let provider: String?
    public let finishReason: String?
    public let usage: OpenRouterUsage?
    public let metadata: OpenRouterMetadata?
}

/// OpenRouter-specific client. Its request surface intentionally exposes OpenRouter routing,
/// reasoning, model-capability, and telemetry features instead of treating it as a generic endpoint.
public struct OpenRouterClient: Sendable {
    public static let chatCompletionsURL = URL(string: "https://openrouter.ai/api/v1/chat/completions")!
    public static let modelsURL = URL(string: "https://openrouter.ai/api/v1/models")!

    /// Fetches full model capabilities needed to form compatible OpenRouter requests.
    public static func fetchModelCatalog(timeout: TimeInterval = 15) async throws -> [OpenRouterModel] {
        var request = URLRequest(url: modelsURL)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await performRequest(
            request,
            timeout: timeout,
            maxRetries: 0
        )
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(OpenRouterModelsResponse.self, from: data)
        return decoded.data.sorted { $0.id.localizedCaseInsensitiveCompare($1.id) == .orderedAscending }
    }

    /// Compatibility helper for callers that only need model IDs.
    public static func fetchModels(timeout: TimeInterval = 15) async throws -> [String] {
        try await fetchModelCatalog(timeout: timeout).map(\.id)
    }

    public static func chatCompletion(
        apiKey: String,
        model: String,
        messages: [ChatMessage],
        systemPrompt: String? = nil,
        temperature: Double? = nil,
        maxTokens: Int? = nil,
        maxCompletionTokens: Int? = nil,
        reasoning: OpenRouterReasoning? = nil,
        provider: OpenRouterProviderPreferences? = nil,
        includeRouterMetadata: Bool = true,
        appReferer: URL? = nil,
        appTitle: String? = nil,
        timeout: TimeInterval = 30
    ) async throws -> OpenRouterCompletion {
        try validateAPIKey(apiKey)

        var allMessages = messages
        if let systemPrompt, !systemPrompt.isEmpty {
            allMessages.insert(.system(systemPrompt), at: 0)
        }

        let body = OpenRouterChatRequest(
            model: model,
            messages: allMessages,
            temperature: temperature,
            maxTokens: maxTokens,
            maxCompletionTokens: maxCompletionTokens,
            reasoning: reasoning,
            provider: provider
        )

        var request = URLRequest(url: chatCompletionsURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        if includeRouterMetadata {
            request.setValue("enabled", forHTTPHeaderField: "X-OpenRouter-Metadata")
        }
        if let appReferer {
            request.setValue(appReferer.absoluteString, forHTTPHeaderField: "HTTP-Referer")
        }
        if let appTitle, !appTitle.isEmpty {
            request.setValue(appTitle, forHTTPHeaderField: "X-OpenRouter-Title")
        }

        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            throw LLMKitError.encodingError
        }

        let (data, response) = try await performRequest(
            request,
            timeout: timeout,
            maxRetries: 0
        )
        try validateOpenRouterResponse(response, data: data)

        let decoded = try decodeJSON(OpenRouterChatResponse.self, from: data)
        guard let choice = decoded.choices.first,
              let text = choice.message.content?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty else {
            throw LLMKitError.noResultReturned
        }

        return OpenRouterCompletion(
            text: text,
            model: decoded.model,
            provider: decoded.provider,
            finishReason: choice.finishReason,
            usage: decoded.usage,
            metadata: decoded.metadata
        )
    }

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
        request.setValue("Bearer \(trimmedKey)", forHTTPHeaderField: "Authorization")

        do {
            let (data, response) = try await performRequest(
                request,
                timeout: timeout,
                maxRetries: 0
            )
            guard let http = response as? HTTPURLResponse else {
                return (false, "No HTTP response received.")
            }
            if (200..<300).contains(http.statusCode) {
                return (true, nil)
            }
            return (false, openRouterErrorMessage(from: data) ?? "HTTP \(http.statusCode)")
        } catch {
            return (false, error.localizedDescription)
        }
    }
}

private struct OpenRouterModelsResponse: Decodable {
    let data: [OpenRouterModel]
}

private struct OpenRouterChatRequest: Encodable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double?
    let stream = false
    let maxTokens: Int?
    let maxCompletionTokens: Int?
    let reasoning: OpenRouterReasoning?
    let provider: OpenRouterProviderPreferences?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case stream
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case reasoning
        case provider
    }
}

private struct OpenRouterChatResponse: Decodable {
    let model: String?
    let provider: String?
    let choices: [Choice]
    let usage: OpenRouterUsage?
    let metadata: OpenRouterMetadata?

    struct Choice: Decodable {
        let message: Message
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case message
            case finishReason = "finish_reason"
        }
    }

    struct Message: Decodable {
        let content: String?
    }

    enum CodingKeys: String, CodingKey {
        case model
        case provider
        case choices
        case usage
        case metadata = "openrouter_metadata"
    }
}

private struct OpenRouterErrorEnvelope: Decodable {
    let error: OpenRouterError

    struct OpenRouterError: Decodable {
        let message: String
    }
}

private func validateOpenRouterResponse(_ response: URLResponse, data: Data) throws {
    guard let http = response as? HTTPURLResponse else {
        throw LLMKitError.networkError("No HTTP response received.")
    }
    guard (200..<300).contains(http.statusCode) else {
        throw LLMKitError.httpError(
            statusCode: http.statusCode,
            message: openRouterErrorMessage(from: data) ?? "No error details"
        )
    }
}

private func openRouterErrorMessage(from data: Data) -> String? {
    try? JSONDecoder().decode(OpenRouterErrorEnvelope.self, from: data).error.message
}
