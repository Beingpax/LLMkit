import Foundation

/// Client for AssemblyAI's asynchronous speech-to-text API.
///
/// Uses the upload -> transcript job -> poll flow for local audio data.
public struct AssemblyAIClient: Sendable {
    private static let apiBase = "https://api.assemblyai.com"

    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        language: String? = nil,
        prompt: String? = nil,
        customVocabulary: [String] = [],
        maxWaitSeconds: TimeInterval = 300,
        timeout: TimeInterval = 30
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let uploadURL = try await uploadAudio(audioData: audioData, apiKey: apiKey, timeout: timeout)
        let transcriptId = try await createTranscript(
            audioURL: uploadURL,
            apiKey: apiKey,
            model: model,
            language: language,
            prompt: prompt,
            customVocabulary: customVocabulary,
            timeout: timeout
        )
        let transcript = try await pollTranscript(
            id: transcriptId,
            apiKey: apiKey,
            maxWaitSeconds: maxWaitSeconds,
            timeout: timeout
        )

        guard !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw LLMKitError.noResultReturned
        }
        return transcript
    }

    public static func verifyAPIKey(_ apiKey: String, timeout: TimeInterval = 10) async -> (isValid: Bool, errorMessage: String?) {
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (false, "API key is missing or empty.")
        }

        guard let url = URL(string: "\(apiBase)/v2/transcript") else {
            return (false, "Invalid URL.")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout
        request.setValue(apiKey, forHTTPHeaderField: "Authorization")

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

    // MARK: - Private

    private static func uploadAudio(audioData: Data, apiKey: String, timeout: TimeInterval) async throws -> String {
        guard let url = URL(string: "\(apiBase)/v2/upload") else {
            throw LLMKitError.invalidURL("\(apiBase)/v2/upload")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "Authorization")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")

        let (data, response) = try await performUpload(request, data: audioData, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(AssemblyAIUploadResponse.self, from: data)
        return decoded.uploadURL
    }

    private static func createTranscript(
        audioURL: String,
        apiKey: String,
        model: String,
        language: String?,
        prompt: String?,
        customVocabulary: [String],
        timeout: TimeInterval
    ) async throws -> String {
        guard let url = URL(string: "\(apiBase)/v2/transcript") else {
            throw LLMKitError.invalidURL("\(apiBase)/v2/transcript")
        }

        let speechModels = speechModels(for: model)
        let primarySpeechModel = speechModels.first ?? model

        var payload: [String: Any] = [
            "audio_url": audioURL,
            "speech_models": speechModels,
            "punctuate": true,
            "format_text": true
        ]

        if let language, !language.isEmpty, language != "auto" {
            payload["language_code"] = language
        } else {
            payload["language_detection"] = true
        }

        let trimmedPrompt = prompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let keyterms = normalizedKeyterms(customVocabulary, model: primarySpeechModel)
        if supportsPrompt(speechModels), !trimmedPrompt.isEmpty {
            payload["prompt"] = appendKeyterms(keyterms, to: trimmedPrompt)
        } else if !keyterms.isEmpty {
            payload["keyterms_prompt"] = keyterms
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        guard let body = try? JSONSerialization.data(withJSONObject: payload) else {
            throw LLMKitError.encodingError
        }
        request.httpBody = body

        let (data, response) = try await performRequest(request, timeout: timeout)
        try validateHTTPResponse(response, data: data)

        let decoded = try decodeJSON(AssemblyAITranscriptCreateResponse.self, from: data)
        return decoded.id
    }

    private static func pollTranscript(
        id: String,
        apiKey: String,
        maxWaitSeconds: TimeInterval,
        timeout: TimeInterval
    ) async throws -> String {
        guard let url = URL(string: "\(apiBase)/v2/transcript/\(id)") else {
            throw LLMKitError.invalidURL("\(apiBase)/v2/transcript/\(id)")
        }

        let start = Date()
        while true {
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = timeout
            request.setValue(apiKey, forHTTPHeaderField: "Authorization")

            let (data, response) = try await URLSession.shared.data(for: request)
            try validateHTTPResponse(response, data: data)

            let decoded = try decodeJSON(AssemblyAITranscriptStatusResponse.self, from: data)
            switch decoded.status.lowercased() {
            case "completed":
                return decoded.text ?? ""
            case "error":
                throw LLMKitError.httpError(statusCode: 500, message: decoded.error ?? "AssemblyAI transcription failed.")
            default:
                break
            }

            if Date().timeIntervalSince(start) > maxWaitSeconds {
                throw LLMKitError.timeout
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    private static func speechModels(for model: String) -> [String] {
        switch model {
        case "universal-3-pro":
            return ["universal-3-pro", "universal-2"]
        case "universal-2":
            return ["universal-2"]
        case "universal-streaming", "universal-streaming-english", "universal-streaming-multilingual", "whisper-rt":
            return ["universal-2"]
        default:
            return [model]
        }
    }

    private static func supportsPrompt(_ speechModels: [String]) -> Bool {
        speechModels.contains("universal-3-pro")
    }

    private static func normalizedKeyterms(_ terms: [String], model: String) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        let limit = model == "universal-2" ? 200 : 1_000
        for term in terms {
            let trimmed = term.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, trimmed.count <= 50 else { continue }
            let key = trimmed.lowercased()
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            result.append(trimmed)
            if result.count == limit { break }
        }
        return result
    }

    private static func appendKeyterms(_ keyterms: [String], to prompt: String) -> String {
        guard !keyterms.isEmpty else { return prompt }
        return "\(prompt)\n\nBoost these terms when they appear in the audio: \(keyterms.joined(separator: ", "))."
    }
}

private struct AssemblyAIUploadResponse: Decodable, Sendable {
    let uploadURL: String

    enum CodingKeys: String, CodingKey {
        case uploadURL = "upload_url"
    }
}

private struct AssemblyAITranscriptCreateResponse: Decodable, Sendable {
    let id: String
}

private struct AssemblyAITranscriptStatusResponse: Decodable, Sendable {
    let status: String
    let text: String?
    let error: String?
}
