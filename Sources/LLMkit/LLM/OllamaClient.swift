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
    ///   - temperature: Sampling temperature (default 0.3).
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The generated response text.
    public static func generate(
        baseURL: URL = defaultBaseURL,
        model: String,
        prompt: String,
        systemPrompt: String,
        temperature: Double = 0.3,
        timeout: TimeInterval = 30
    ) async throws -> String {
        let url = baseURL.appendingPathComponent("api/generate")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "system": systemPrompt,
            "temperature": temperature,
            "stream": false
        ]

        guard let bodyData = try? JSONSerialization.data(withJSONObject: body) else {
            throw LLMKitError.encodingError
        }
        request.httpBody = bodyData

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(OllamaGenerateResponse.self, from: data)
        return decoded.response
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
