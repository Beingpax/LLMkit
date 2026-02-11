import Foundation

/// Client for the ElevenLabs speech-to-text REST API.
///
/// Sends audio as multipart/form-data to ElevenLabs' `/v1/speech-to-text` endpoint.
public struct ElevenLabsClient: Sendable {

    /// Transcribes audio data using the ElevenLabs API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes.
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: ElevenLabs API key.
    ///   - model: Model name (e.g. `"scribe_v1"`, `"scribe_v2"`).
    ///   - language: Optional language code. Pass `nil` for auto-detect.
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        language: String? = nil,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let url = URL(string: "https://api.elevenlabs.io/v1/speech-to-text")!

        var form = MultipartFormData()

        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)
        form.addField(name: "model_id", value: model)
        form.addField(name: "temperature", value: "0.0")
        form.addField(name: "tag_audio_events", value: "false")

        if let language, !language.isEmpty {
            form.addField(name: "language_code", value: language)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)

        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(ElevenLabsResponse.self, from: data)
        return decoded.text
    }

    /// Verifies that an ElevenLabs API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: ElevenLabs API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.elevenlabs.io/v1/user")!)
        request.timeoutInterval = timeout
        request.setValue(apiKey, forHTTPHeaderField: "xi-api-key")

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

private struct ElevenLabsResponse: Decodable, Sendable {
    let text: String
}
