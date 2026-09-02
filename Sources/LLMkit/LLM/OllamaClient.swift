import Foundation

/// Client for the local Ollama API.
///
/// Ollama runs locally (default `http://localhost:11434`) and provides both a generate API
/// (`/api/generate`) and an OpenAI-compatible chat completions API.
///
/// For chat completions, you can also use `OpenAILLMClient` with `{baseURL}/v1/chat/completions`.
public struct OllamaClient: Sendable {

    /// The default Ollama server URL.
    public static let defaultBaseURL = URL(string: "http://localhost:11434")!

    /// Checks whether the Ollama server is reachable.
    ///
    /// - Parameters:
    ///   - baseURL: The Ollama server base URL (default `http://localhost:11434`).
    ///   - timeout: Request timeout in seconds (default 5).
    /// - Returns: `true` if the server is reachable and responds with 2xx.
    public static func checkConnection(baseURL: URL = defaultBaseURL, timeout: TimeInterval = 5) async -> Bool {
        var request = URLRequest(url: baseURL)
        request.timeoutInterval = timeout
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200..<300).contains(http.statusCode)
        } catch {
            return false
        }
    }

    /// Fetches the list of available models from the Ollama server.
    ///
    /// - Parameters:
    ///   - baseURL: The Ollama server base URL.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: An array of `OllamaModel` objects.
    public static func fetchModels(baseURL: URL = defaultBaseURL, timeout: TimeInterval = 10) async throws -> [OllamaModel] {
        let url = baseURL.appendingPathComponent("api/tags")

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(OllamaModelsResponse.self, from: data)
        return decoded.models
    }

    /// Generates a response using Ollama's generate API (`/api/generate`).
    ///
    /// - Parameters:
    ///   - baseURL: The Ollama server base URL.
    ///   - model: Model name (e.g. `"llama2"`, `"mistral"`).
    ///   - prompt: The user prompt.
    ///   - systemPrompt: The system prompt.
    ///   - options: Optional per-request model parameters. Nil values are omitted so
    ///     Ollama can use the model or Modelfile defaults.
    ///   - think: Optional native Ollama thinking control. Use `false` to disable thinking.
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The generated response text.
    public static func generate(
        baseURL: URL = defaultBaseURL,
        model: String,
        prompt: String,
        systemPrompt: String,
        options: OllamaGenerationOptions? = nil,
        think: Bool? = nil,
        timeout: TimeInterval = 30
    ) async throws -> String {
        let request = try makeGenerateRequest(
            baseURL: baseURL,
            model: model,
            prompt: prompt,
            systemPrompt: systemPrompt,
            options: options,
            think: think
        )

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(OllamaGenerateResponse.self, from: data)
        return decoded.response
    }

    /// Compatibility overload for callers that previously supplied a temperature directly.
    @available(*, deprecated, message: "Use the options parameter instead")
    public static func generate(
        baseURL: URL = defaultBaseURL,
        model: String,
        prompt: String,
        systemPrompt: String,
        temperature: Double,
        think: Bool? = nil,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try await generate(
            baseURL: baseURL,
            model: model,
            prompt: prompt,
            systemPrompt: systemPrompt,
            options: OllamaGenerationOptions(temperature: temperature),
            think: think,
            timeout: timeout
        )
    }

    static func makeGenerateRequest(
        baseURL: URL,
        model: String,
        prompt: String,
        systemPrompt: String,
        options: OllamaGenerationOptions?,
        think: Bool?
    ) throws -> URLRequest {
        let url = baseURL.appendingPathComponent("api/generate")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try JSONEncoder().encode(
                OllamaGenerateRequest(
                    model: model,
                    prompt: prompt,
                    system: systemPrompt,
                    options: options?.isEmpty == false ? options : nil,
                    stream: false,
                    think: think
                )
            )
        } catch {
            throw LLMKitError.encodingError
        }

        return request
    }
}

/// Optional per-request model parameters for Ollama's native generate API.
public struct OllamaGenerationOptions: Codable, Sendable, Equatable {
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    var isEmpty: Bool {
        temperature == nil && topP == nil && topK == nil
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
    }
}

// MARK: - Public Models

/// Represents a model available on the Ollama server.
public struct OllamaModel: Codable, Sendable, Identifiable {
    public let name: String
    public let modified_at: String
    public let size: Int64
    public let digest: String
    public let details: ModelDetails

    public var id: String { name }

    public struct ModelDetails: Codable, Sendable {
        public let format: String
        public let family: String
        public let families: [String]?
        public let parameter_size: String
        public let quantization_level: String
    }
}

// MARK: - Private Response Models

private struct OllamaModelsResponse: Decodable, Sendable {
    let models: [OllamaModel]
}

private struct OllamaGenerateResponse: Decodable, Sendable {
    let response: String
}

private struct OllamaGenerateRequest: Encodable, Sendable {
    let model: String
    let prompt: String
    let system: String
    let options: OllamaGenerationOptions?
    let stream: Bool
    let think: Bool?
}
