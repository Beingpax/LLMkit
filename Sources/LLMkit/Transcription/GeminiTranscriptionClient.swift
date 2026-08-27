import Foundation

/// Output cleanup mode used by Gemini's dedicated transcription models.
public enum GeminiTranscriptionMode: String, Sendable {
    /// Preserves filler words, repetitions, and false starts.
    case verbatim
    /// Removes disfluencies and applies intent-aware formatting.
    case smart
}

/// Client for the Gemini (Google AI) speech-to-text REST API.
///
/// General Gemini models use `generateContent`. The dedicated
/// `gemini-3.5-transcribe` model uses the Interactions API and Files API.
public struct GeminiTranscriptionClient: Sendable {

    /// Transcribes audio data using the Gemini API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes.
    ///   - apiKey: Google AI / Gemini API key.
    ///   - model: Model name (e.g. `"gemini-2.5-flash"`, `"gemini-2.5-pro"`).
    ///   - mimeType: MIME type of the audio (default `"audio/wav"`).
    ///   - fileName: Display name used when uploading to the Gemini Files API.
    ///   - language: Optional BCP-47 language hint. Pass `nil` or `"auto"` for detection.
    ///   - customVocabulary: Terms used to bias dedicated speech recognition.
    ///   - mode: Dedicated transcription cleanup mode (default `verbatim`).
    ///   - timeout: Request timeout in seconds (default 60).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        apiKey: String,
        model: String,
        mimeType: String = "audio/wav",
        fileName: String = "audio.wav",
        language: String? = nil,
        customVocabulary: [String] = [],
        mode: GeminiTranscriptionMode = .verbatim,
        timeout: TimeInterval = 60
    ) async throws -> String {
        try validateAPIKey(apiKey)

        if model.lowercased() == "gemini-3.5-transcribe" {
            return try await transcribeWithDedicatedModel(
                audioData: audioData,
                apiKey: apiKey,
                model: model,
                mimeType: mimeType,
                fileName: fileName,
                language: language,
                customVocabulary: customVocabulary,
                mode: mode,
                timeout: timeout
            )
        }

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
            ],
            generationConfig: generationConfig(for: model)
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

    private static func generationConfig(for model: String) -> GeminiTranscriptionGenerationConfig? {
        if model.lowercased() == "gemini-3.7-flash" {
            return GeminiTranscriptionGenerationConfig(
                thinkingConfig: GeminiTranscriptionThinkingConfig(thinkingLevel: "low")
            )
        }
        return nil
    }

    private static func transcribeWithDedicatedModel(
        audioData: Data,
        apiKey: String,
        model: String,
        mimeType: String,
        fileName: String,
        language: String?,
        customVocabulary: [String],
        mode: GeminiTranscriptionMode,
        timeout: TimeInterval
    ) async throws -> String {
        let uploadedFile = try await uploadFile(
            audioData: audioData,
            apiKey: apiKey,
            mimeType: mimeType,
            fileName: fileName,
            timeout: timeout
        )

        do {
            let result = try await createTranscriptionInteraction(
                apiKey: apiKey,
                model: model,
                uploadedFile: uploadedFile,
                language: language,
                customVocabulary: customVocabulary,
                mode: mode,
                timeout: timeout
            )
            await deleteFile(uploadedFile, apiKey: apiKey, timeout: timeout)
            return result
        } catch {
            await deleteFile(uploadedFile, apiKey: apiKey, timeout: timeout)
            throw error
        }
    }

    private static func uploadFile(
        audioData: Data,
        apiKey: String,
        mimeType: String,
        fileName: String,
        timeout: TimeInterval
    ) async throws -> GeminiUploadedFile {
        let startURLString = "https://generativelanguage.googleapis.com/upload/v1beta/files"
        guard let startURL = URL(string: startURLString) else {
            throw LLMKitError.invalidURL(startURLString)
        }

        var startRequest = URLRequest(url: startURL)
        startRequest.httpMethod = "POST"
        startRequest.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        startRequest.setValue("resumable", forHTTPHeaderField: "X-Goog-Upload-Protocol")
        startRequest.setValue("start", forHTTPHeaderField: "X-Goog-Upload-Command")
        startRequest.setValue(String(audioData.count), forHTTPHeaderField: "X-Goog-Upload-Header-Content-Length")
        startRequest.setValue(mimeType, forHTTPHeaderField: "X-Goog-Upload-Header-Content-Type")
        startRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        startRequest.httpBody = try encodeJSON(GeminiFileUploadStartRequest(file: .init(displayName: fileName)))

        let (startData, startResponse) = try await performRequest(startRequest, timeout: timeout)
        try validateHTTPResponse(startResponse, data: startData)

        guard let httpResponse = startResponse as? HTTPURLResponse,
              let uploadURLString = httpResponse.value(forHTTPHeaderField: "X-Goog-Upload-URL"),
              let uploadURL = URL(string: uploadURLString) else {
            throw LLMKitError.networkError("Gemini Files API did not return an upload URL.")
        }

        var uploadRequest = URLRequest(url: uploadURL)
        uploadRequest.httpMethod = "POST"
        uploadRequest.setValue(String(audioData.count), forHTTPHeaderField: "Content-Length")
        uploadRequest.setValue("0", forHTTPHeaderField: "X-Goog-Upload-Offset")
        uploadRequest.setValue("upload, finalize", forHTTPHeaderField: "X-Goog-Upload-Command")

        let (uploadData, uploadResponse) = try await performUpload(
            uploadRequest,
            data: audioData,
            timeout: timeout
        )
        try validateHTTPResponse(uploadResponse, data: uploadData)
        return try decodeJSON(GeminiFileUploadResponse.self, from: uploadData).file
    }

    private static func createTranscriptionInteraction(
        apiKey: String,
        model: String,
        uploadedFile: GeminiUploadedFile,
        language: String?,
        customVocabulary: [String],
        mode: GeminiTranscriptionMode,
        timeout: TimeInterval
    ) async throws -> String {
        let endpointString = "https://generativelanguage.googleapis.com/v1beta/interactions"
        guard let endpoint = URL(string: endpointString) else {
            throw LLMKitError.invalidURL(endpointString)
        }

        let languageCodes: [String]
        if let language, !language.isEmpty, language.lowercased() != "auto" {
            languageCodes = [language]
        } else {
            languageCodes = []
        }

        let body = GeminiDedicatedTranscriptionRequest(
            model: model,
            input: [
                GeminiDedicatedAudioInput(
                    type: "audio",
                    uri: uploadedFile.uri,
                    mimeType: uploadedFile.mimeType
                )
            ],
            store: false,
            generationConfig: GeminiDedicatedGenerationConfig(
                transcriptionConfig: GeminiDedicatedTranscriptionConfig(
                    languageCodes: languageCodes,
                    customVocabulary: normalizedVocabulary(customVocabulary),
                    mode: GeminiDedicatedTranscriptionMode(type: mode.rawValue)
                )
            )
        )

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encodeJSON(body)

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(GeminiDedicatedTranscriptionResponse.self, from: data)
        let text = decoded.steps
            .filter { $0.type == "model_output" }
            .flatMap(\.content)
            .filter { $0.type == "text" }
            .compactMap(\.text)
            .joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return text
    }

    private static func deleteFile(
        _ file: GeminiUploadedFile,
        apiKey: String,
        timeout: TimeInterval
    ) async {
        let endpointString = "https://generativelanguage.googleapis.com/v1beta/\(file.name)"
        guard let endpoint = URL(string: endpointString) else { return }

        var request = URLRequest(url: endpoint)
        request.httpMethod = "DELETE"
        request.setValue(apiKey, forHTTPHeaderField: "x-goog-api-key")

        guard let (data, response) = try? await performRequest(request, timeout: timeout, maxRetries: 0) else {
            return
        }
        try? validateHTTPResponse(response, data: data)
    }

    private static func normalizedVocabulary(_ terms: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []

        for term in terms {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let key = trimmed.lowercased()
            guard seen.insert(key).inserted else { continue }
            result.append(trimmed)
            if result.count == 1_000 { break }
        }
        return result
    }

    private static func encodeJSON<T: Encodable>(_ value: T) throws -> Data {
        do {
            return try JSONEncoder().encode(value)
        } catch {
            throw LLMKitError.encodingError
        }
    }
}

// MARK: - Request Models

private struct GeminiRequest: Encodable, Sendable {
    let contents: [GeminiContent]
    let generationConfig: GeminiTranscriptionGenerationConfig?
}

private struct GeminiTranscriptionGenerationConfig: Encodable, Sendable {
    let thinkingConfig: GeminiTranscriptionThinkingConfig
}

private struct GeminiTranscriptionThinkingConfig: Encodable, Sendable {
    let thinkingLevel: String
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

// MARK: - Dedicated Transcription Models

private struct GeminiFileUploadStartRequest: Encodable, Sendable {
    let file: FileMetadata

    struct FileMetadata: Encodable, Sendable {
        let displayName: String

        enum CodingKeys: String, CodingKey {
            case displayName = "display_name"
        }
    }
}

private struct GeminiFileUploadResponse: Decodable, Sendable {
    let file: GeminiUploadedFile
}

private struct GeminiUploadedFile: Decodable, Sendable {
    let name: String
    let uri: String
    let mimeType: String
}

private struct GeminiDedicatedTranscriptionRequest: Encodable, Sendable {
    let model: String
    let input: [GeminiDedicatedAudioInput]
    let store: Bool
    let generationConfig: GeminiDedicatedGenerationConfig

    enum CodingKeys: String, CodingKey {
        case model, input, store
        case generationConfig = "generation_config"
    }
}

private struct GeminiDedicatedAudioInput: Encodable, Sendable {
    let type: String
    let uri: String
    let mimeType: String

    enum CodingKeys: String, CodingKey {
        case type, uri
        case mimeType = "mime_type"
    }
}

private struct GeminiDedicatedGenerationConfig: Encodable, Sendable {
    let transcriptionConfig: GeminiDedicatedTranscriptionConfig

    enum CodingKeys: String, CodingKey {
        case transcriptionConfig = "transcription_config"
    }
}

private struct GeminiDedicatedTranscriptionConfig: Encodable, Sendable {
    let languageCodes: [String]
    let customVocabulary: [String]
    let mode: GeminiDedicatedTranscriptionMode

    enum CodingKeys: String, CodingKey {
        case languageCodes = "language_codes"
        case customVocabulary = "custom_vocabulary"
        case mode
    }
}

private struct GeminiDedicatedTranscriptionMode: Encodable, Sendable {
    let type: String
}

private struct GeminiDedicatedTranscriptionResponse: Decodable, Sendable {
    let steps: [Step]

    struct Step: Decodable, Sendable {
        let type: String
        let content: [Content]
    }

    struct Content: Decodable, Sendable {
        let type: String
        let text: String?
    }
}
