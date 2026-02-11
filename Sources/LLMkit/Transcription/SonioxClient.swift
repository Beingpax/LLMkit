import Foundation

/// Client for the Soniox speech-to-text REST API.
///
/// Uses a multi-step flow: upload file → create transcription job → poll status → fetch transcript.
public struct SonioxClient: Sendable {
    private static let apiBase = "https://api.soniox.com/v1"

    /// Transcribes audio data using the Soniox API.
    ///
    /// This method handles the full async flow: file upload, job creation, status polling, and
    /// transcript retrieval.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes.
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: Soniox API key.
    ///   - model: Model name (e.g. `"stt-async-v4"`).
    ///   - language: Optional language hint. Pass `nil` for auto-detect.
    ///   - customVocabulary: Optional list of custom terms to boost recognition.
    ///   - maxWaitSeconds: Maximum seconds to wait for transcription completion (default 300).
    ///   - timeout: Per-request timeout in seconds (default 30).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        language: String? = nil,
        customVocabulary: [String] = [],
        maxWaitSeconds: TimeInterval = 300,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let fileId = try await uploadFile(audioData: audioData, fileName: fileName, apiKey: apiKey, timeout: timeout)
        let transcriptionId = try await createTranscription(
            fileId: fileId,
            apiKey: apiKey,
            model: model,
            language: language,
            customVocabulary: customVocabulary,
            timeout: timeout
        )
        try await pollTranscriptionStatus(id: transcriptionId, apiKey: apiKey, maxWaitSeconds: maxWaitSeconds, timeout: timeout)
        let transcript = try await fetchTranscript(id: transcriptionId, apiKey: apiKey, timeout: timeout)

        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return transcript
    }

    /// Verifies that a Soniox API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: Soniox API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        guard let url = URL(string: "\(apiBase)/files") else { return (false, "Invalid URL.") }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

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

    // MARK: - Private Steps

    private static func uploadFile(audioData: Data, fileName: String, apiKey: String, timeout: TimeInterval) async throws -> String {
        guard let url = URL(string: "\(apiBase)/files") else {
            throw LLMKitError.invalidURL("\(apiBase)/files")
        }

        var form = MultipartFormData()
        form.addFile(name: "file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(FileUploadResponse.self, from: data)
        return decoded.id
    }

    private static func createTranscription(
        fileId: String,
        apiKey: String,
        model: String,
        language: String?,
        customVocabulary: [String],
        timeout: TimeInterval
    ) async throws -> String {
        guard let url = URL(string: "\(apiBase)/transcriptions") else {
            throw LLMKitError.invalidURL("\(apiBase)/transcriptions")
        }

        var payload: [String: Any] = [
            "file_id": fileId,
            "model": model,
            "enable_speaker_diarization": false
        ]

        if !customVocabulary.isEmpty {
            payload["context"] = ["terms": customVocabulary]
        }

        if let language, !language.isEmpty {
            payload["language_hints"] = [language]
            payload["language_hints_strict"] = true
            payload["enable_language_identification"] = true
        } else {
            payload["enable_language_identification"] = true
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            throw LLMKitError.encodingError
        }
        request.httpBody = body

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(CreateTranscriptionResponse.self, from: data)
        return decoded.id
    }

    private static func pollTranscriptionStatus(id: String, apiKey: String, maxWaitSeconds: TimeInterval, timeout: TimeInterval) async throws {
        guard let url = URL(string: "\(apiBase)/transcriptions/\(id)") else {
            throw LLMKitError.invalidURL("\(apiBase)/transcriptions/\(id)")
        }

        let start = Date()
        while true {
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = timeout
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

            let (data, response) = try await URLSession.shared.data(for: request)
            try validateHTTPResponse(response, data: data)

            if let status = try? JSONDecoder().decode(TranscriptionStatusResponse.self, from: data) {
                switch status.status.lowercased() {
                case "completed":
                    return
                case "failed":
                    throw LLMKitError.httpError(statusCode: 500, message: "Soniox transcription job failed.")
                default:
                    break
                }
            }

            if Date().timeIntervalSince(start) > maxWaitSeconds {
                throw LLMKitError.timeout
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    private static func fetchTranscript(id: String, apiKey: String, timeout: TimeInterval) async throws -> String {
        guard let url = URL(string: "\(apiBase)/transcriptions/\(id)/transcript") else {
            throw LLMKitError.invalidURL("\(apiBase)/transcriptions/\(id)/transcript")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        if let decoded = try? JSONDecoder().decode(TranscriptResponse.self, from: data) {
            return decoded.text
        }
        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return text
        }
        throw LLMKitError.noResultReturned
    }
}

// MARK: - Response Models

private struct FileUploadResponse: Decodable, Sendable { let id: String }
private struct CreateTranscriptionResponse: Decodable, Sendable { let id: String }
private struct TranscriptionStatusResponse: Decodable, Sendable { let status: String }
private struct TranscriptResponse: Decodable, Sendable { let text: String }
