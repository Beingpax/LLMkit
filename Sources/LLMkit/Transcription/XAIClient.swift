import Foundation

/// Client for the xAI speech-to-text REST API.
///
/// Sends audio as multipart/form-data to `https://api.x.ai/v1/stt`.
public struct XAIClient: Sendable {

    /// Transcribes audio data using the xAI STT API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (any supported format — API detects from content).
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: xAI API key.
    ///   - language: Optional BCP-47 language code. Pass `nil` for auto-detect.
    ///   - format: Whether to apply text formatting (Inverse Text Normalization). Requires `language`.
    ///   - timeout: Request timeout in seconds (default 60).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        language: String? = nil,
        format: Bool = false,
        timeout: TimeInterval = 60
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let url = URL(string: "https://api.x.ai/v1/stt")!

        var form = MultipartFormData()

        if let language, !language.isEmpty, language != "auto" {
            form.addField(name: "language", value: language)
            if format {
                form.addField(name: "format", value: "true")
            }
        }

        // Per xAI docs, the `file` field must be the last field in the multipart body.
        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)

        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(XAITranscriptionResponse.self, from: data)
        return decoded.text
    }

    /// Verifies that an xAI API key is valid by calling the `/v1/api-key` endpoint.
    ///
    /// - Parameters:
    ///   - apiKey: xAI API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.x.ai/v1/api-key")!)
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

private struct XAITranscriptionResponse: Decodable, Sendable {
    let text: String
}
