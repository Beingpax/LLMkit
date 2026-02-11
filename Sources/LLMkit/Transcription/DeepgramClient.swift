import Foundation

/// Client for the Deepgram speech-to-text REST API.
///
/// Sends raw audio data (binary) to Deepgram's `/v1/listen` endpoint
/// and returns the transcribed text.
public struct DeepgramClient: Sendable {

    /// Transcribes audio data using the Deepgram API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV/PCM format).
    ///   - apiKey: Deepgram API key.
    ///   - model: Model name (e.g. `"nova-3"`, `"nova-3-medical"`).
    ///   - language: Optional BCP-47 language code. Pass `nil` for auto-detect.
    ///   - smartFormat: Enable smart formatting (default `true`).
    ///   - punctuate: Enable punctuation (default `true`).
    ///   - paragraphs: Enable paragraph detection (default `true`).
    ///   - customVocabulary: Optional list of custom keywords to boost recognition.
    ///   - timeout: Request timeout in seconds (default 30).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        apiKey: String,
        model: String,
        language: String? = nil,
        smartFormat: Bool = true,
        punctuate: Bool = true,
        paragraphs: Bool = true,
        customVocabulary: [String] = [],
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        var queryItems = [
            URLQueryItem(name: "model", value: model),
            URLQueryItem(name: "smart_format", value: smartFormat ? "true" : "false"),
            URLQueryItem(name: "punctuate", value: punctuate ? "true" : "false"),
            URLQueryItem(name: "paragraphs", value: paragraphs ? "true" : "false")
        ]

        if let language, !language.isEmpty {
            queryItems.append(URLQueryItem(name: "language", value: language))
        }

        for term in customVocabulary {
            queryItems.append(URLQueryItem(name: "keyterm", value: term))
        }

        var components = URLComponents(string: "https://api.deepgram.com/v1/listen")!
        components.queryItems = queryItems

        guard let url = components.url else {
            throw LLMKitError.invalidURL("https://api.deepgram.com/v1/listen")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Token \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("audio/wav", forHTTPHeaderField: "Content-Type")

        let (data, response) = try await performUpload(request, data: audioData, timeout: timeout)

        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(DeepgramResponse.self, from: data)
        guard let transcript = decoded.results.channels.first?.alternatives.first?.transcript,
              !transcript.isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return transcript
    }

    /// Verifies that a Deepgram API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: Deepgram API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        var request = URLRequest(url: URL(string: "https://api.deepgram.com/v1/projects")!)
        request.timeoutInterval = timeout
        request.setValue("Token \(apiKey)", forHTTPHeaderField: "Authorization")

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

private struct DeepgramResponse: Decodable, Sendable {
    let results: Results

    struct Results: Decodable, Sendable {
        let channels: [Channel]

        struct Channel: Decodable, Sendable {
            let alternatives: [Alternative]

            struct Alternative: Decodable, Sendable {
                let transcript: String
                let confidence: Double?
            }
        }
    }
}
