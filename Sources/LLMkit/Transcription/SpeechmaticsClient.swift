import Foundation

/// Client for the Speechmatics batch speech-to-text REST API.
///
/// Uses a multi-step flow: submit job → poll status → fetch transcript.
/// API docs: https://docs.speechmatics.com/api-ref/batch/speechmatics-asr-rest-api
public struct SpeechmaticsClient: Sendable {
    private static let apiBase = "https://asr.api.speechmatics.com/v2"

    /// Transcribes audio data using the Speechmatics batch API.
    ///
    /// - Parameters:
    ///   - audioData: Raw audio bytes (WAV format).
    ///   - fileName: Name of the audio file (e.g. `"recording.wav"`).
    ///   - apiKey: Speechmatics API key.
    ///   - language: Language code (e.g. `"en"`). Pass `nil` or `"auto"` for auto-detect.
    ///   - operatingPoint: Operating point (`"enhanced"` or `"standard"`). Default `"enhanced"`.
    ///   - customVocabulary: Optional list of custom terms to boost recognition.
    ///   - maxWaitSeconds: Maximum seconds to wait for transcription completion (default 300).
    ///   - timeout: Per-request timeout in seconds (default 30).
    /// - Returns: The transcribed text.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        language: String? = nil,
        operatingPoint: String = "enhanced",
        customVocabulary: [String] = [],
        maxWaitSeconds: TimeInterval = 300,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let jobId = try await submitJob(
            audioData: audioData,
            fileName: fileName,
            apiKey: apiKey,
            language: language,
            operatingPoint: operatingPoint,
            customVocabulary: customVocabulary,
            timeout: timeout
        )
        try await pollJobStatus(id: jobId, apiKey: apiKey, maxWaitSeconds: maxWaitSeconds, timeout: timeout)
        let transcript = try await fetchTranscript(id: jobId, apiKey: apiKey, timeout: timeout)

        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return transcript
    }

    /// Verifies that a Speechmatics API key is valid.
    ///
    /// - Parameters:
    ///   - apiKey: Speechmatics API key.
    ///   - timeout: Request timeout in seconds (default 10).
    /// - Returns: A tuple of (isValid, errorMessage). `errorMessage` is `nil` on success.
    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        guard let url = URL(string: "\(apiBase)/jobs") else { return (false, "Invalid URL.") }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
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

    // MARK: - Private Steps

    private static func submitJob(
        audioData: Data,
        fileName: String,
        apiKey: String,
        language: String?,
        operatingPoint: String,
        customVocabulary: [String],
        timeout: TimeInterval
    ) async throws -> String {
        guard let url = URL(string: "\(apiBase)/jobs") else {
            throw LLMKitError.invalidURL("\(apiBase)/jobs")
        }

        // Build the config JSON
        let lang = mapLanguage(language)
        var transcriptionConfig: [String: Any] = [
            "language": lang,
            "operating_point": operatingPoint
        ]

        if !customVocabulary.isEmpty {
            transcriptionConfig["additional_vocab"] = customVocabulary.map { ["content": $0] }
        }

        let config: [String: Any] = [
            "type": "transcription",
            "transcription_config": transcriptionConfig
        ]

        guard let configData = try? JSONSerialization.data(withJSONObject: config),
              let configString = String(data: configData, encoding: .utf8) else {
            throw LLMKitError.encodingError
        }

        var form = MultipartFormData()
        form.addField(name: "config", value: configString)
        form.addFile(name: "data_file", fileName: fileName, mimeType: "audio/wav", fileData: audioData)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(form.contentType, forHTTPHeaderField: "Content-Type")

        let (data, response) = try await performUpload(request, data: form.data, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(JobSubmitResponse.self, from: data)
        return decoded.id
    }

    private static func pollJobStatus(id: String, apiKey: String, maxWaitSeconds: TimeInterval, timeout: TimeInterval) async throws {
        guard let url = URL(string: "\(apiBase)/jobs/\(id)") else {
            throw LLMKitError.invalidURL("\(apiBase)/jobs/\(id)")
        }

        let start = Date()
        while true {
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = timeout
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

            let (data, response) = try await URLSession.shared.data(for: request)
            try validateHTTPResponse(response, data: data)

            if let status = try? JSONDecoder().decode(JobStatusResponse.self, from: data) {
                switch status.job.status.lowercased() {
                case "done":
                    return
                case "rejected":
                    throw LLMKitError.httpError(statusCode: 500, message: "Speechmatics transcription job was rejected.")
                case "deleted":
                    throw LLMKitError.httpError(statusCode: 410, message: "Speechmatics transcription job was deleted.")
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
        guard let url = URL(string: "\(apiBase)/jobs/\(id)/transcript?format=txt") else {
            throw LLMKitError.invalidURL("\(apiBase)/jobs/\(id)/transcript")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        if let text = String(data: data, encoding: .utf8), !text.isEmpty {
            return text
        }
        throw LLMKitError.noResultReturned
    }

    /// Maps VoiceInk language codes to Speechmatics language codes.
    private static func mapLanguage(_ language: String?) -> String {
        guard let language, !language.isEmpty, language != "auto" else { return "auto" }
        switch language {
        case "zh": return "cmn"
        default: return language
        }
    }
}

// MARK: - Response Models

private struct JobSubmitResponse: Decodable, Sendable { let id: String }
private struct JobStatusResponse: Decodable, Sendable {
    let job: Job
    struct Job: Decodable, Sendable { let status: String }
}
