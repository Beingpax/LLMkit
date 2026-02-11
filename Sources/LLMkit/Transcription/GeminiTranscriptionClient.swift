import Foundation

/// Client for the Gemini (Google AI) speech-to-text REST API.
///
/// Sends base64-encoded audio inline in a JSON body to Gemini's `generateContent` endpoint.
public struct GeminiTranscriptionClient: Sendable {

    /// Transcribes audio data using the Gemini API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes.
    ///   - apiKey: Google AI / Gemini API key.
    ///   - model: Model name (e.g. `"gemini-2.5-flash"`, `"gemini-2.5-pro"`).
    ///   - mimeType: MIME type of the audio (default `"audio/wav"`).
    ///   - timeout: Request timeout in seconds (default 60).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        apiKey: String,
        model: String,
        mimeType: String = "audio/wav",
        timeout: TimeInterval = 60
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let urlString = "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent"
        guard let url = URL(string: urlString) else {
            throw LLMKitError.invalidURL(urlString)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

        let base64Audio = audioData.base64EncodedString()

        let requestBody = GeminiRequest(
            contents: [
                GeminiContent(parts: [
                    GeminiPart(text: "Please transcribe this audio file. Provide only the transcribed text.", inlineData: nil),
                    GeminiPart(text: nil, inlineData: GeminiInlineData(mimeType: mimeType, data: base64Audio))
                ])
            ]
        )

        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
        } catch {
            throw LLMKitError.encodingError
        }

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(GeminiResponse.self, from: data)
        guard let candidate = decoded.candidates.first,
              let part = candidate.content.parts.first,
              !part.text.isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return part.text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Verifies that a Gemini API key is valid by making a lightweight models list request.
    ///
    /// - Parameters:
    ///   - apiKey: Gemini API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        guard let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/models") else {
            return (false, "Invalid URL.")
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

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

// MARK: - Request Models

private struct GeminiRequest: Encodable, Sendable {
    let contents: [GeminiContent]
}

private struct GeminiContent: Encodable, Sendable {
    let parts: [GeminiPart]
}

private struct GeminiPart: Encodable, Sendable {
    let text: String?
    let inlineData: GeminiInlineData?

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let text { try container.encode(text, forKey: .text) }
        if let inlineData { try container.encode(inlineData, forKey: .inlineData) }
    }

    private enum CodingKeys: String, CodingKey {
        case text, inlineData
    }
}

private struct GeminiInlineData: Encodable, Sendable {
    let mimeType: String
    let data: String
}

// MARK: - Response Models

private struct GeminiResponse: Decodable, Sendable {
    let candidates: [GeminiCandidate]
}

private struct GeminiCandidate: Decodable, Sendable {
    let content: GeminiResponseContent
}

private struct GeminiResponseContent: Decodable, Sendable {
    let parts: [GeminiResponsePart]
}

private struct GeminiResponsePart: Decodable, Sendable {
    let text: String
}
