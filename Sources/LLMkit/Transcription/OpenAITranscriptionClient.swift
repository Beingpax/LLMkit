import Foundation

/// Client for OpenAI-compatible speech-to-text REST APIs.
///
/// Works with any provider that implements the OpenAI `/v1/audio/transcriptions` endpoint,
/// including Groq, Mistral (via OpenAI format), and custom endpoints.
public struct OpenAITranscriptionClient: Sendable {

    /// Transcribes audio using an OpenAI-compatible transcription endpoint.
    ///
    /// - Parameters:
    ///   - baseURL: Provider base URL (e.g. `https://api.groq.com/openai`).
    ///     The path `/v1/audio/transcriptions` is appended automatically.
    ///   - audioData: Raw audio bytes.
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: API key for the provider.
    ///   - model: Model name (e.g. `"whisper-large-v3-turbo"`).
    ///   - language: Optional language code. Pass `nil` for auto-detect.
    ///   - prompt: Optional transcription prompt/hint.
    ///   - timeout: Request timeout in seconds (default 60).
    /// - Returns: The transcribed text.
    public static func transcribe(
        baseURL: URL,
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        language: String? = nil,
        prompt: String? = nil,
        timeout: TimeInterval = 60
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let url = baseURL.appendingPathComponent("v1/audio/transcriptions")

        var form = MultipartFormData()

        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)
        form.addField(name: "model", value: model)

        if let language, !language.isEmpty {
            form.addField(name: "language", value: language)
        }

        if let prompt, !prompt.isEmpty {
            form.addField(name: "prompt", value: prompt)
        }

        form.addField(name: "response_format", value: "json")
        form.addField(name: "temperature", value: "0")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)

        try validateHTTPResponse(response, data: data)

        // Try structured JSON decode first, fall back to raw string
        if let decoded = try? JSONDecoder().decode(OpenAITranscriptionResponse.self, from: data),
           let text = decoded.text {
            return text
        }
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return text
        }
        throw LLMKitError.noResultReturned
    }

    /// Verifies that an API key is valid against an OpenAI-compatible provider.
    ///
    /// - Parameters:
    ///   - baseURL: Provider base URL (e.g. `https://api.groq.com/openai`).
    ///   - apiKey: API key for the provider.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(baseURL: URL, apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("v1/models"))
        request.timeoutInterval = timeout
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

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

// MARK: - Response Models

private struct OpenAITranscriptionResponse: Decodable, Sendable {
    let text: String?
}
