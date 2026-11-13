import Foundation

/// Shared constants and wire-protocol helpers for Zoom AI Services Scribe (Fast and Live).
/// API docs: https://developers.zoom.us/docs/ai-services/scribe/
/// Models and languages: https://developers.zoom.us/docs/ai-services/scribe/speech-configuration/
enum ZoomScribe {
    static let liveURL = URL(string: "wss://api.zoom.us/v2/aiservices/scribe/live")!
    static let fastURL = URL(string: "https://api.zoom.us/v2/aiservices/scribe/transcribe")!
    static let liveSubprotocol = "live-asr"

    static let multilingualModel = "zoom-scribe-multilingual"
    static let englishModel = "zoom-scribe-en"
    static let englishProModel = "zoom-scribe-en-pro"
    static let models: [String] = [multilingualModel, englishModel, englishProModel]

    /// Scribe expects a regional locale. Base codes outside this table are passed through unchanged.
    static let localeByLanguage: [String: String] = [
        "en": "en-US", "ar": "ar-SA", "zh": "zh-CN", "fr": "fr-FR", "de": "de-DE",
        "it": "it-IT", "ja": "ja-JP", "ko": "ko-KR", "pt": "pt-BR", "es": "es-ES",
    ]

    static let vocabularyLimit = 100

    static let defaultTrackingIdPrefix = "zoom_ai_community_llmkit_"

    /// Zoom uses the tracking id for support lookups, so every request gets a fresh one.
    static func makeTrackingId(prefix: String = defaultTrackingIdPrefix) -> String {
        prefix + UUID().uuidString
    }

    /// Appends the tracking id that Zoom support uses to find a request.
    static func withTrackingId(_ message: String, _ trackingId: String) -> String {
        "\(message) (tracking_id=\(trackingId))"
    }

    /// Text for a server `error` event, shared by the handshake and runtime paths.
    static func serverErrorMessage(code: String, message: String, trackingId: String) -> String {
        withTrackingId("Zoom Scribe error \(code): \(message)", trackingId)
    }

    /// Detection only exists on the multilingual model. Other models keep the en-US fallback.
    static func usesLanguageDetection(language: String?, model: String?) -> Bool {
        model == multilingualModel && isAutoLanguage(language)
    }

    /// Text polishing is an opt-in add-on. It only works for en-US and Scribe rejects it with language detection
    /// (HTTP 400), so it is sent only when both conditions hold. Otherwise nothing is sent.
    static func polishingEnabled(requested: Bool, model: String?, language: String?) -> Bool {
        requested
            && !usesLanguageDetection(language: language, model: model)
            && locale(for: language).lowercased() == "en-us"
    }

    /// The top-level `outputs` object that enables text polishing. It must sit next to `config`, not inside it.
    static func polishingOutputs() -> [String: Any] {
        ["text_polishing": ["enabled": true]]
    }

    /// Returns the polished text with speaker and channel tags removed, or nil when Scribe reports a
    /// `[polishing_failed: ...]` marker, in which case the caller uses the unpolished transcript.
    /// Tags are stripped because this output is used as dictation text, where `[Speaker 1]` would be pasted in.
    static func polishedOutput(from text: String) -> String? {
        guard !text.contains("[polishing_failed:") else { return nil }
        let cleaned = text
            .replacingOccurrences(of: #"\[(?:Speaker \d+|channel\d+)\]\s*"#, with: " ", options: .regularExpression)
            .replacingOccurrences(of: #"[ \t]{2,}"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? nil : cleaned
    }

    /// Removes the spaces that `zoom-scribe-zh` puts between Chinese characters and around Chinese punctuation.
    /// Spaces are kept when neither neighbor is a CJK character or CJK punctuation, and every other model is unchanged.
    static func cleanedTranscript(_ text: String, model: String?) -> String {
        guard model == "zoom-scribe-zh" else { return text }
        return text.replacingOccurrences(
            of: #"(?<=[\p{Han}　-〿＀-￯])[ 　]+|[ 　]+(?=[\p{Han}　-〿＀-￯])"#,
            with: "", options: .regularExpression)
    }

    /// Base codes that have a dedicated `zoom-scribe-<code>` model. Other codes use the multilingual model.
    static let dedicatedModelBaseCodes: Set<String> = ["ar", "zh", "fr", "de", "it", "ja", "pt", "es"]

    /// The model to send to Zoom. For `zoom-scribe-en`, English stays on that model, codes with a dedicated
    /// model switch to `zoom-scribe-<code>`, and other codes (such as ko) use the multilingual model.
    /// Other input models are returned unchanged.
    static func apiModel(for model: String?, language: String?) -> String? {
        guard model == englishModel else {
            return model
        }
        let baseCode = String(locale(for: language).prefix { $0 != "-" }).lowercased()
        if baseCode == "en" {
            return englishModel
        }
        if dedicatedModelBaseCodes.contains(baseCode) {
            return "zoom-scribe-" + baseCode
        }
        return multilingualModel
    }

    /// Maps a base language code to a Scribe locale. Codes that already contain a region pass through unchanged.
    /// Missing or `auto` input maps to en-US. Unknown base codes pass through so Zoom can reject them.
    static func locale(for language: String?) -> String {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if trimmed.contains("-") {
            return trimmed
        }
        if isAutoLanguage(language) {
            return "en-US"
        }
        return localeByLanguage[trimmed.lowercased()] ?? trimmed
    }

    /// Returns the Fast `audio/*` or `video/*` MIME type for a documented container, or nil if undocumented.
    static func fastMimeType(forFileName fileName: String) -> String? {
        switch URL(fileURLWithPath: fileName).pathExtension.lowercased() {
        case "wav": return "audio/wav"
        case "mp3": return "audio/mpeg"
        case "m4a": return "audio/mp4"
        case "mp4": return "video/mp4"
        case "webm": return "audio/webm"
        default: return nil
        }
    }

    // MARK: - Internal protocol helpers

    static let sessionCloseMessage = #"{"type":"session.close"}"#

    static func sessionUpdateMessage(
        language: String?, model: String?, vocabulary: [String], textPolishing: Bool = false
    ) -> String {
        let detectsLanguage = usesLanguageDetection(language: language, model: model)
        var config: [String: Any] = ["interim_results": true]
        if let apiModel = apiModel(for: model, language: language) {
            config["model"] = apiModel
        }
        // With detection enabled, Scribe reports the language per turn, so no language is sent.
        if detectsLanguage {
            config["language_detection"] = ["mode": "continuous"]
        } else {
            config["language"] = locale(for: language)
        }
        let phrases = normalizedVocabulary(vocabulary)
        if !phrases.isEmpty {
            config["vocabulary"] = ["phrases": phrases]
        }
        var payload: [String: Any] = [
            "type": "session.update",
            "audio": ["format": "pcm16"],
            "config": config,
        ]
        if polishingEnabled(requested: textPolishing, model: model, language: language) {
            payload["outputs"] = polishingOutputs()
        }
        return String(decoding: (try? JSONSerialization.data(withJSONObject: payload)) ?? Data(), as: UTF8.self)
    }

    enum ServerEvent: Equatable {
        case sessionUpdated(model: String?)
        case speechStarted
        /// The full hypothesis of the in-progress turn. It replaces, not appends to, the previous delta.
        case delta(String)
        /// The final text of a completed turn.
        case transcript(String)
        case error(code: String, message: String, fatal: Bool)
        case sessionClosed
        case other
    }

    /// Returns nil for non-JSON frames or frames without a `type`.
    static func parseServerEvent(_ text: String) -> ServerEvent? {
        guard let data = text.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = object["type"] as? String
        else { return nil }

        switch type {
        case "session.updated":
            return .sessionUpdated(model: object["model"] as? String)
        case "input_audio_buffer.speech_started":
            return .speechStarted
        case "transcription.delta":
            return .delta(object["delta"] as? String ?? "")
        case "transcription.completed":
            // Polished text replaces the transcript only when it is usable. Deltas are never polished.
            let transcript = object["transcript"] as? String ?? ""
            let polished = (object["outputs"] as? [String: Any])
                .flatMap { $0["text_polishing"] as? [String: Any] }
                .flatMap { $0["text"] as? String }
                .flatMap(polishedOutput(from:))
            return .transcript(polished ?? transcript)
        case "error":
            // Codes arrive as numbers or strings depending on the failure.
            let error = object["error"] as? [String: Any] ?? [:]
            let code = error["code"].map { "\($0)" } ?? "unknown"
            let message = error["message"] as? String ?? (object["error"] as? String) ?? "unknown error"
            return .error(code: code, message: message, fatal: error["fatal"] as? Bool ?? false)
        case "session.closed":
            return .sessionClosed
        default:
            return .other
        }
    }

    static func makeLiveRequest(apiKey: String, trackingId: String) -> URLRequest {
        var request = URLRequest(url: liveURL)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue(liveSubprotocol, forHTTPHeaderField: "Sec-WebSocket-Protocol")
        request.setValue(trackingId, forHTTPHeaderField: "x-zm-trackingid")
        return request
    }

    /// Trims, drops empty entries, and caps the list at `vocabularyLimit`.
    static func normalizedVocabulary(_ terms: [String]) -> [String] {
        let trimmed = terms
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return Array(trimmed.prefix(vocabularyLimit))
    }

    private static func isAutoLanguage(_ language: String?) -> Bool {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty || trimmed.lowercased() == "auto"
    }
}
