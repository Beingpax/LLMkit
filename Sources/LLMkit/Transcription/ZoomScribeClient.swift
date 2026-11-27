import Foundation

/// Client for Zoom AI Services Scribe Fast mode, which transcribes a complete file in one HTTP request.
/// API docs: https://developers.zoom.us/docs/ai-services/scribe/fast-mode/
public struct ZoomScribeClient: Sendable {

    /// Transcribes an audio file using Zoom Scribe Fast mode.
    ///
    /// - Parameters:
    ///   - audioData: Encoded audio or video in one of WAV, MP3, M4A, MP4 or WebM.
    ///   - fileName: Name of the file. Its extension selects the container (e.g. `"recording.wav"`).
    ///   - apiKey: Zoom AI Services API key.
    ///   - model: Scribe model (e.g. `"zoom-scribe-multilingual"`).
    ///   - language: Base language code (e.g. `"en"`). Pass `nil` or `"auto"` to detect the language
    ///     on the multilingual model.
    ///   - customVocabulary: Phrases to boost. Trimmed, empty entries dropped, capped at 100.
    ///   - trackingIdPrefix: Prefix for the per-request tracking id, which is sent to Zoom and appended to errors.
    ///   - timeout: Request timeout in seconds (default 60).
    ///   - textPolishing: Opt-in add-on that returns polished text. Applies only to en-US without language
    ///     detection. Other requests ignore it. The unpolished transcript is used when polishing fails.
    /// - Returns: The transcribed text. Empty when the audio contains no speech.
    public static func transcribe(
        audioData: Data,
        fileName: String = "audio.wav",
        apiKey: String,
        model: String,
        language: String? = nil,
        customVocabulary: [String] = [],
        trackingIdPrefix: String = "zoom_ai_community_llmkit_",
        timeout: TimeInterval = 60,
        textPolishing: Bool = false
    ) async throws -> String {
        try validateAPIKey(apiKey)

        let trackingId = ZoomScribe.makeTrackingId(prefix: trackingIdPrefix)
        let body = try prepareRequestBody(
            audioData: audioData, fileName: fileName, model: model, language: language,
            vocabulary: customVocabulary, trackingId: trackingId, textPolishing: textPolishing)

        var request = URLRequest(url: ZoomScribe.fastURL)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(trackingId, forHTTPHeaderField: "x-zm-trackingid")

        let result: (Data, URLResponse)
        do {
            result = try await performUpload(request, data: body, timeout: timeout)
        } catch {
            throw transportError(error, trackingId: trackingId)
        }
        let (data, response) = result

        guard let http = response as? HTTPURLResponse else {
            throw LLMKitError.networkError(ZoomScribe.withTrackingId("No HTTP response received.", trackingId))
        }
        guard (200..<300).contains(http.statusCode) else {
            throw LLMKitError.httpError(
                statusCode: http.statusCode,
                message: ZoomScribe.withTrackingId(errorMessage(statusCode: http.statusCode, data: data), trackingId))
        }
        guard let transcript = parseTranscript(from: data) else {
            throw LLMKitError.noResultReturned
        }
        let routedModel = ZoomScribe.apiModel(for: model, language: language)
        if ZoomScribe.polishingEnabled(requested: textPolishing, model: model, language: language),
           let polished = parsePolishedTranscript(from: data) {
            return ZoomScribe.cleanedTranscript(polished, model: routedModel)
        }
        return ZoomScribe.cleanedTranscript(transcript, model: routedModel)
    }

    // MARK: - Internal helpers

    /// Checks the file before anything is sent and returns the encoded request body.
    static func prepareRequestBody(
        audioData: Data, fileName: String, model: String, language: String?, vocabulary: [String], trackingId: String,
        textPolishing: Bool = false
    ) throws -> Data {
        guard let mimeType = ZoomScribe.fastMimeType(forFileName: fileName) else {
            let extensionName = URL(fileURLWithPath: fileName).pathExtension
            let subject = extensionName.isEmpty ? "files without an extension" : ".\(extensionName) files"
            throw LLMKitError.httpError(
                statusCode: 415,
                message: ZoomScribe.withTrackingId("Zoom Scribe does not accept \(subject)", trackingId))
        }
        let body = try makeRequestBody(
            audioData: audioData, mimeType: mimeType, model: model, language: language, vocabulary: vocabulary,
            textPolishing: textPolishing)
        return body
    }

    private static func makeRequestBody(
        audioData: Data, mimeType: String, model: String, language: String?, vocabulary: [String],
        textPolishing: Bool
    ) throws -> Data {
        var config: [String: Any] = ["model": ZoomScribe.apiModel(for: model, language: language) ?? model]
        if ZoomScribe.usesLanguageDetection(language: language, model: model) {
            config["language_detection"] = ["mode": "continuous"]
        } else {
            config["language"] = ZoomScribe.locale(for: language)
        }
        let phrases = ZoomScribe.normalizedVocabulary(vocabulary)
        if !phrases.isEmpty {
            config["vocabulary"] = ["phrases": phrases]
        }
        var payload: [String: Any] = [
            "file": "data:\(mimeType);base64,\(audioData.base64EncodedString())",
            "config": config,
        ]
        // `outputs` is top-level, next to `config`. Inside `config`, Zoom ignores it.
        if ZoomScribe.polishingEnabled(requested: textPolishing, model: model, language: language) {
            payload["outputs"] = ZoomScribe.polishingOutputs()
        }
        do {
            return try JSONSerialization.data(withJSONObject: payload)
        } catch {
            throw LLMKitError.encodingError
        }
    }

    /// Adds the tracking id to transport failures. Timeouts and cancellation keep their own meaning.
    private static func transportError(_ error: Error, trackingId: String) -> Error {
        if error is CancellationError {
            return error
        }
        switch error {
        case LLMKitError.timeout:
            return LLMKitError.timeout
        case LLMKitError.httpError(let statusCode, let message):
            return LLMKitError.httpError(statusCode: statusCode, message: ZoomScribe.withTrackingId(message, trackingId))
        case LLMKitError.networkError(let detail):
            return LLMKitError.networkError(ZoomScribe.withTrackingId(detail, trackingId))
        case is LLMKitError:
            return error
        default:
            return LLMKitError.networkError(ZoomScribe.withTrackingId(error.localizedDescription, trackingId))
        }
    }

    /// Reads `result.text_display`, falling back to joining `result.segments[].text`. Returns nil when neither exists.
    /// An empty `text_display` is a valid result for silent audio and returns "".
    static func parseTranscript(from data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = object["result"] as? [String: Any]
        else { return nil }

        let rawText: String
        if let textDisplay = result["text_display"] as? String {
            rawText = textDisplay
        } else if let segments = result["segments"] as? [[String: Any]] {
            rawText = segments.compactMap { $0["text"] as? String }.joined(separator: " ")
        } else {
            return nil
        }
        return rawText
    }

    /// Reads `outputs.text_polishing.text`. Returns nil when it is absent or contains a failure marker.
    static func parsePolishedTranscript(from data: Data) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let outputs = object["outputs"] as? [String: Any],
              let polishing = outputs["text_polishing"] as? [String: Any],
              let text = polishing["text"] as? String
        else { return nil }
        return ZoomScribe.polishedOutput(from: text)
    }

    /// Prefers the JSON `message` field. HTML or empty bodies (for example gateway error pages) are never surfaced raw.
    static func errorMessage(statusCode: Int, data: Data) -> String {
        if statusCode == 413 {
            return "The audio is too large for a single Zoom Scribe request"
        }
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let message = object["message"] as? String,
           !message.isEmpty {
            return message
        }
        let raw = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty || raw.hasPrefix("<") {
            return "HTTP \(statusCode)"
        }
        return String(raw.prefix(500))
    }
}
