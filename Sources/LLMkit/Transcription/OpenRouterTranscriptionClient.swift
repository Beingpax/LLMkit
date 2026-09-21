import Foundation

/// OpenRouter's speech-to-text endpoint. Language is omitted so OpenRouter auto-detects it.
public struct OpenRouterTranscriptionClient: Sendable {
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        timeout: TimeInterval = 60
    ) async throws -> String {
        try validateAPIKey(apiKey)

        var form = MultipartFormData()
        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)
        form.addField(name: "model", value: model)

        var request = URLRequest(url: URL(string: "https://openrouter.ai/api/v1/audio/transcriptions")!)
        request.httpMethod = "POST"
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let result = try decodeJSON(TranscriptionResponse.self, from: data)
        guard let text = result.text, !text.isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return text
    }

    private struct TranscriptionResponse: Decodable {
        let text: String?
    }

}
