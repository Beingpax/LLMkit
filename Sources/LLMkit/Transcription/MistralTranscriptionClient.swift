import Foundation

/// Client for the Mistral speech-to-text REST API.
///
/// Sends audio as multipart/form-data to Mistral's `/v1/audio/transcriptions` endpoint.
/// Uses the `x-api-key` header (not Bearer token).
public struct MistralTranscriptionClient: Sendable {

    /// Transcribes audio data using the Mistral API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes.
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: Mistral API key.
    ///   - model: Model name (e.g. `"voxtral-mini-latest"`).
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        guard let url = URL(string: "https://api.mistral.ai/v1/audio/transcriptions") else {
            throw LLMKitError.invalidURL("https://api.mistral.ai/v1/audio/transcriptions")
        }

        var form = MultipartFormData()

        form.addField(name: "model", value: model)
        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)

        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(MistralTranscriptionResponse.self, from: data)
        return decoded.text
    }

    /// Verifies that a Mistral API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: Mistral API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.mistral.ai/v1/models")!)
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

private struct MistralTranscriptionResponse: Decodable, Sendable {
    let text: String
}
