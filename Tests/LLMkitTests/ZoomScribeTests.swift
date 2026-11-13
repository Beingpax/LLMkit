import Foundation
import Testing

@testable import LLMkit

struct ZoomScribeTests {
    @Test func mapsBaseLanguageCodesToScribeLocales() {
        #expect(ZoomScribe.locale(for: "en") == "en-US")
        #expect(ZoomScribe.locale(for: "pt") == "pt-BR")
        #expect(ZoomScribe.locale(for: "fr-CA") == "fr-CA")
    }

    @Test func unknownOrMissingLanguageFallsBackToEnglish() {
        #expect(ZoomScribe.locale(for: nil) == "en-US")
        #expect(ZoomScribe.locale(for: "") == "en-US")
        #expect(ZoomScribe.locale(for: "auto") == "en-US")
        #expect(ZoomScribe.locale(for: "ru") == "en-US")
        #expect(ZoomScribe.locale(for: "yue") == "en-US")
    }

    @Test func sessionUpdateCarriesModelAndInterimResults() throws {
        let message = ZoomScribe.sessionUpdateMessage(
            language: "en", model: ZoomScribe.multilingualModel, vocabulary: ["Zoom"])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(object["type"] as? String == "session.update")
        #expect(Set(object.keys) == ["type", "audio", "config"])
        #expect(object["language"] == nil)
        #expect(config["language"] as? String == "en-US")
        #expect(config["model"] as? String == ZoomScribe.multilingualModel)
        #expect(config["interim_results"] as? Bool == true)
        let vocabulary = try #require(config["vocabulary"] as? [String: Any])
        #expect(vocabulary["phrases"] as? [String] == ["Zoom"])
    }

    @Test func sessionUpdateOmitsModelWhenNil() throws {
        let message = ZoomScribe.sessionUpdateMessage(language: nil, model: nil, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["model"] == nil)
        #expect(config["vocabulary"] == nil)
    }

    @Test func autoLanguageOnMultilingualModelEnablesDetection() throws {
        let message = ZoomScribe.sessionUpdateMessage(
            language: "auto", model: ZoomScribe.multilingualModel, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(object["language"] == nil)
        #expect(config["language"] == nil)
        let detection = try #require(config["language_detection"] as? [String: Any])
        #expect(detection["mode"] as? String == "continuous")
        #expect(config["model"] as? String == ZoomScribe.multilingualModel)
    }

    @Test func explicitLanguageOnMultilingualModelSkipsDetection() throws {
        let message = ZoomScribe.sessionUpdateMessage(
            language: "ja", model: ZoomScribe.multilingualModel, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(object["language"] == nil)
        #expect(config["language"] as? String == "ja-JP")
        #expect(config["language_detection"] == nil)
    }

    @Test func autoLanguageOnEnProKeepsEnglishFallback() throws {
        let message = ZoomScribe.sessionUpdateMessage(language: "auto", model: ZoomScribe.englishProModel, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(object["language"] == nil)
        #expect(config["language"] as? String == "en-US")
        #expect(config["language_detection"] == nil)
    }

    @Test func autoLanguageOnEnglishModelKeepsEnglishWithoutDetection() throws {
        let message = ZoomScribe.sessionUpdateMessage(language: "auto", model: ZoomScribe.englishModel, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(object["language"] == nil)
        #expect(config["language"] as? String == "en-US")
        #expect(config["language_detection"] == nil)
    }

    @Test func vocabularyIsTrimmedFilteredAndCappedAt100() throws {
        let terms = ["  Zoom  ", "", "   "] + (0..<150).map { "term\($0)" }
        let message = ZoomScribe.sessionUpdateMessage(language: "en", model: nil, vocabulary: terms)
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        let vocabulary = try #require(config["vocabulary"] as? [String: Any])
        let phrases = try #require(vocabulary["phrases"] as? [String])
        #expect(phrases.count == ZoomScribe.vocabularyLimit)
        #expect(phrases.first == "Zoom")
        #expect(phrases.last == "term98")
    }

    @Test func usesLanguageDetectionOnlyForAutoOnMultilingual() {
        #expect(ZoomScribe.usesLanguageDetection(language: nil, model: ZoomScribe.multilingualModel))
        #expect(ZoomScribe.usesLanguageDetection(language: " AUTO ", model: ZoomScribe.multilingualModel))
        #expect(!ZoomScribe.usesLanguageDetection(language: "en", model: ZoomScribe.multilingualModel))
        #expect(!ZoomScribe.usesLanguageDetection(language: nil, model: ZoomScribe.englishModel))
        #expect(!ZoomScribe.usesLanguageDetection(language: nil, model: nil))
    }

    @Test func apiModelRoutesDedicatedLanguagesOnEnglishModelToPerLanguageModel() {
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "zh") == "zoom-scribe-zh")
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "ja") == "zoom-scribe-ja")
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "es") == "zoom-scribe-es")
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "ja-JP") == "zoom-scribe-ja")
    }

    @Test func apiModelRoutesKoreanOnEnglishModelToMultilingual() {
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "ko") == ZoomScribe.multilingualModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "ko-KR") == ZoomScribe.multilingualModel)
    }

    @Test func apiModelKeepsEnglishModelForEnglishOrMissingLanguage() {
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "en") == ZoomScribe.englishModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "en-US") == ZoomScribe.englishModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: nil) == ZoomScribe.englishModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishModel, language: "auto") == ZoomScribe.englishModel)
    }

    @Test func apiModelLeavesOtherModelsUnchanged() {
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishProModel, language: "en") == ZoomScribe.englishProModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.englishProModel, language: "zh") == ZoomScribe.englishProModel)
        #expect(ZoomScribe.apiModel(for: ZoomScribe.multilingualModel, language: "zh") == ZoomScribe.multilingualModel)
        #expect(ZoomScribe.apiModel(for: nil, language: "zh") == nil)
    }

    @Test func sessionUpdateRoutesDedicatedLanguageOnEnglishModelToPerLanguageModel() throws {
        let message = ZoomScribe.sessionUpdateMessage(
            language: "zh", model: ZoomScribe.englishModel, vocabulary: [])
        let object = try #require(try JSONSerialization.jsonObject(with: Data(message.utf8)) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["model"] as? String == "zoom-scribe-zh")
        #expect(config["language"] as? String == "zh-CN")
        #expect(config["language_detection"] == nil)
    }

    @Test func fastRoutesDedicatedLanguageOnEnglishModelToPerLanguageModel() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.englishModel,
            language: "zh", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["model"] as? String == "zoom-scribe-zh")
        #expect(config["language"] as? String == "zh-CN")
        #expect(config["language_detection"] == nil)
    }

    @Test func fastRoutesKoreanOnEnglishModelToMultilingualWithLocale() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.englishModel,
            language: "ko", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["model"] as? String == ZoomScribe.multilingualModel)
        #expect(config["language"] as? String == "ko-KR")
        #expect(config["language_detection"] == nil)
    }

    @Test func fastExplicitLanguageOnMultilingualSendsLocaleWithoutDetection() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.multilingualModel,
            language: "zh", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["model"] as? String == ZoomScribe.multilingualModel)
        #expect(config["language"] as? String == "zh-CN")
        #expect(config["language_detection"] == nil)
    }

    @Test func fastRequestBodyHasLocaleModelAndFile() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1, 2, 3]), fileName: "call.wav", model: ZoomScribe.englishProModel,
            language: "en", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(object["file"] as? String == "data:audio/wav;base64,AQID")
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["language"] as? String == "en-US")
        #expect(config["model"] as? String == ZoomScribe.englishProModel)
        #expect(config["vocabulary"] == nil)
    }

    @Test func fastRequestBodyIncludesVocabularyPhrases() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data(), fileName: "memo.m4a", model: ZoomScribe.multilingualModel,
            language: "fr", vocabulary: ["  Zoom Scribe ", " "], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["language"] as? String == "fr-FR")
        let vocabulary = try #require(config["vocabulary"] as? [String: Any])
        #expect(vocabulary["phrases"] as? [String] == ["Zoom Scribe"])
    }

    @Test func fastAutoLanguageOnMultilingualUsesDetectionConfig() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.multilingualModel,
            language: nil, vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["language"] == nil)
        let detection = try #require(config["language_detection"] as? [String: Any])
        #expect(detection["mode"] as? String == "continuous")
    }

    @Test func fastAutoLanguageOnEnProFallsBackToEnglish() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.englishProModel,
            language: "auto", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        let config = try #require(object["config"] as? [String: Any])
        #expect(config["language"] as? String == "en-US")
        #expect(config["language_detection"] == nil)
    }

    @Test func fastParsesTextDisplay() {
        let data = Data(#"{"result":{"text_display":"Hello there.","segments":[{"text":"ignored"}]}}"#.utf8)
        #expect(ZoomScribeClient.parseTranscript(from: data) == "Hello there.")
    }

    @Test func fastFallsBackToSegmentText() {
        let data = Data(#"{"result":{"segments":[{"text":"Hi"},{"text":"there"}]}}"#.utf8)
        #expect(ZoomScribeClient.parseTranscript(from: data) == "Hi there")
    }

    @Test func fastEmptyTextDisplayIsEmptyResult() {
        let data = Data(#"{"result":{"text_display":""}}"#.utf8)
        #expect(ZoomScribeClient.parseTranscript(from: data) == "")
    }

    @Test func fastRejectsResponseWithoutTranscript() {
        #expect(ZoomScribeClient.parseTranscript(from: Data("{}".utf8)) == nil)
    }

    @Test func errorMessageUsesJSONMessageField() {
        let data = Data(#"{"code":124,"message":"Invalid Access token"}"#.utf8)
        #expect(ZoomScribeClient.errorMessage(statusCode: 401, data: data) == "Invalid Access token")
    }

    @Test func errorMessageNeverSurfacesHTMLFor413() {
        let html = Data("<html>\r\n<head><title>413 Request Entity Too Large</title></head>\r\n</html>".utf8)
        #expect(
            ZoomScribeClient.errorMessage(statusCode: 413, data: html)
                == "The audio is too large for a single Zoom Scribe request")
    }

    @Test func tooLargeResponseIsReportedWithTrackingId() {
        let html = Data("<html><body>413 Request Entity Too Large</body></html>".utf8)
        let message = ZoomScribe.withTrackingId(ZoomScribeClient.errorMessage(statusCode: 413, data: html), "trk")
        #expect(message == "The audio is too large for a single Zoom Scribe request (tracking_id=trk)")
    }

    @Test func errorMessageFallsBackToStatusForHTMLAndEmptyBodies() {
        let html = Data("<html><body>Bad Gateway</body></html>".utf8)
        #expect(ZoomScribeClient.errorMessage(statusCode: 502, data: html) == "HTTP 502")
        #expect(ZoomScribeClient.errorMessage(statusCode: 500, data: Data()) == "HTTP 500")
    }

    @Test func errorMessageTruncatesPlainTextBodies() {
        let long = Data(String(repeating: "x", count: 900).utf8)
        #expect(ZoomScribeClient.errorMessage(statusCode: 400, data: long).count == 500)
    }

    @Test func fastMimeTypeMapsDocumentedContainersOnly() {
        #expect(ZoomScribe.fastMimeType(forFileName: "call.wav") == "audio/wav")
        #expect(ZoomScribe.fastMimeType(forFileName: "CALL.MP3") == "audio/mpeg")
        #expect(ZoomScribe.fastMimeType(forFileName: "memo.m4a") == "audio/mp4")
        #expect(ZoomScribe.fastMimeType(forFileName: "clip.mp4") == "video/mp4")
        #expect(ZoomScribe.fastMimeType(forFileName: "clip.webm") == "audio/webm")
        #expect(ZoomScribe.fastMimeType(forFileName: "audio.flac") == nil)
        #expect(ZoomScribe.fastMimeType(forFileName: "noextension") == nil)
    }

    @Test func fastBodyCarriesMimeTypeDerivedFromFileName() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "clip.MP4", model: ZoomScribe.multilingualModel,
            language: "en", vocabulary: [], trackingId: "trk")
        let object = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect(object["file"] as? String == "data:video/mp4;base64,AQ==")
    }

    @Test func fastRejectsUndocumentedContainerBeforeSending() {
        let rejection = rejectedRequest {
            try ZoomScribeClient.prepareRequestBody(
                audioData: Data([1]), fileName: "audio.flac", model: ZoomScribe.multilingualModel,
                language: nil, vocabulary: [], trackingId: "trk")
        }
        #expect(rejection?.statusCode == 415)
        #expect(rejection?.message == "Zoom Scribe does not accept .flac files (tracking_id=trk)")
    }

    @Test func fastRejectsFileWithoutExtension() {
        let rejection = rejectedRequest {
            try ZoomScribeClient.prepareRequestBody(
                audioData: Data([1]), fileName: "recording", model: ZoomScribe.multilingualModel,
                language: nil, vocabulary: [], trackingId: "trk")
        }
        #expect(rejection?.statusCode == 415)
        #expect(rejection?.message == "Zoom Scribe does not accept files without an extension (tracking_id=trk)")
    }

    @Test func fastLargeBodyIsSentWithoutLocalCap() throws {
        let body = try ZoomScribeClient.prepareRequestBody(
            audioData: Data(count: 40_000_000), fileName: "call.wav", model: ZoomScribe.multilingualModel,
            language: nil, vocabulary: [], trackingId: "trk")
        #expect(body.count > 50_000_000)
    }

    /// Returns the HTTP status and message of an `LLMKitError.httpError` thrown by `body`, or nil.
    private func rejectedRequest(_ body: () throws -> Data) -> (statusCode: Int, message: String)? {
        do {
            _ = try body()
            return nil
        } catch LLMKitError.httpError(let statusCode, let message) {
            return (statusCode, message)
        } catch {
            return nil
        }
    }

    @Test func parsesSessionUpdatedWithAndWithoutModel() {
        #expect(ZoomScribe.parseServerEvent(#"{"type":"session.updated","model":"zoom-scribe-en"}"#)
            == .sessionUpdated(model: "zoom-scribe-en"))
        #expect(ZoomScribe.parseServerEvent(#"{"type":"session.updated"}"#) == .sessionUpdated(model: nil))
    }

    @Test func parsesSpeechStartedDeltaAndTranscript() {
        #expect(ZoomScribe.parseServerEvent(#"{"type":"input_audio_buffer.speech_started"}"#) == .speechStarted)
        #expect(ZoomScribe.parseServerEvent(#"{"type":"transcription.delta","delta":"Hel"}"#) == .delta("Hel"))
        #expect(ZoomScribe.parseServerEvent(#"{"type":"transcription.completed","transcript":"Hello."}"#)
            == .transcript("Hello."))
    }

    @Test func parsesErrorWithStringAndNumericCodes() {
        #expect(ZoomScribe.parseServerEvent(
            #"{"type":"error","error":{"code":"quota","message":"Out","fatal":true}}"#)
            == .error(code: "quota", message: "Out", fatal: true))
        #expect(ZoomScribe.parseServerEvent(
            #"{"type":"error","error":{"code":4001,"message":"Bad model"}}"#)
            == .error(code: "4001", message: "Bad model", fatal: false))
    }

    @Test func parsesSessionClosedAndUnknownTypes() {
        #expect(ZoomScribe.parseServerEvent(#"{"type":"session.closed"}"#) == .sessionClosed)
        #expect(ZoomScribe.parseServerEvent(#"{"type":"session.created"}"#) == .other)
        #expect(ZoomScribe.parseServerEvent("not json") == nil)
        #expect(ZoomScribe.parseServerEvent(#"{"no":"type"}"#) == nil)
    }

    @Test func sessionCloseMessageIsExact() {
        #expect(ZoomScribe.sessionCloseMessage == #"{"type":"session.close"}"#)
    }

    @Test func liveRequestCarriesProtocolHeaders() {
        let request = ZoomScribe.makeLiveRequest(apiKey: "key", trackingId: "trk")
        #expect(request.url == ZoomScribe.liveURL)
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer key")
        #expect(request.value(forHTTPHeaderField: "Sec-WebSocket-Protocol") == "live-asr")
        #expect(request.value(forHTTPHeaderField: "x-zm-trackingid") == "trk")
    }

    @Test func trackingIdUsesPrefixAndIsUnique() {
        let first = ZoomScribe.makeTrackingId()
        #expect(first.hasPrefix(ZoomScribe.defaultTrackingIdPrefix))
        #expect(first != ZoomScribe.makeTrackingId())
    }

    @Test func serverErrorMessageIncludesCodeAndTrackingId() {
        #expect(ZoomScribe.serverErrorMessage(code: "HTTP 401", message: "Invalid", trackingId: "abc")
            == "Zoom Scribe error HTTP 401: Invalid (tracking_id=abc)")
    }

    @Test func keyCheckReportsUnauthorizedAsInvalidKey() {
        let id = "zoom_ai_community_llmkit_test"
        for status in [401, 403] {
            let error = LLMKitError.httpError(
                statusCode: status, message: ZoomScribe.withTrackingId("Zoom Scribe refused the connection", id))
            #expect(ZoomScribeStreamingClient.verificationFailureMessage(for: error, trackingId: id)
                == "Invalid Zoom Scribe API key")
        }
    }

    @Test func keyCheckDropsTrackingIdFromOtherFailures() {
        let id = "zoom_ai_community_llmkit_test"
        let serverError = LLMKitError.httpError(
            statusCode: 503, message: ZoomScribe.withTrackingId("Zoom Scribe refused the connection", id))
        #expect(ZoomScribeStreamingClient.verificationFailureMessage(for: serverError, trackingId: id)
            == "HTTP 503: Zoom Scribe refused the connection")
        let networkError = LLMKitError.networkError(ZoomScribe.withTrackingId("The network connection was lost.", id))
        #expect(ZoomScribeStreamingClient.verificationFailureMessage(for: networkError, trackingId: id)
            == "Network error: The network connection was lost.")
        #expect(ZoomScribeStreamingClient.verificationFailureMessage(for: LLMKitError.timeout, trackingId: id)
            == "The request timed out.")
    }

    @Test func polishingIsEnabledOnlyForEnglishWithoutLanguageDetection() {
        #expect(ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.englishModel, language: "en"))
        #expect(ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.englishProModel, language: "en-US"))
        #expect(ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.multilingualModel, language: "en"))
        #expect(!ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.multilingualModel, language: nil))
        #expect(!ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.multilingualModel, language: "auto"))
        #expect(!ZoomScribe.polishingEnabled(requested: true, model: ZoomScribe.englishModel, language: "zh"))
        #expect(!ZoomScribe.polishingEnabled(requested: false, model: ZoomScribe.englishModel, language: "en"))
    }

    @Test func polishedOutputRemovesSpeakerAndChannelTagsAndTrims() {
        let text = "  [Speaker 1] Hello there. [channel0] Hi [Speaker 12] bye [channel1]Ok  "
        #expect(ZoomScribe.polishedOutput(from: text) == "Hello there. Hi bye Ok")
    }

    @Test func polishedOutputKeepsWordsApartWhenTagIsRemoved() {
        #expect(ZoomScribe.polishedOutput(from: "Hello.[Speaker 2] Bye") == "Hello. Bye")
        #expect(ZoomScribe.polishedOutput(from: "[Speaker 1] Welcome everyone. [Speaker 2] Let's get started.") == "Welcome everyone. Let's get started.")
        #expect(ZoomScribe.polishedOutput(from: "[channel0] Hi [channel1] there") == "Hi there")
    }

    @Test func polishedOutputIsNilWhenPolishingFailed() {
        #expect(ZoomScribe.polishedOutput(from: "Hello [polishing_failed: timeout]") == nil)
        #expect(ZoomScribe.polishedOutput(from: "[polishing_failed: partial] hi") == nil)
        #expect(ZoomScribe.polishedOutput(from: "[Speaker 1] Hi [polishing_failed: timeout]") == nil)
    }

    @Test func fastPolishingAddsTopLevelOutputsOnlyWhenEnabled() throws {
        let on = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.englishModel,
            language: "en", vocabulary: [], trackingId: "trk", textPolishing: true)
        let onObject = try #require(try JSONSerialization.jsonObject(with: on) as? [String: Any])
        let outputs = try #require(onObject["outputs"] as? [String: Any])
        let polishing = try #require(outputs["text_polishing"] as? [String: Any])
        #expect(polishing["enabled"] as? Bool == true)
        let onConfig = try #require(onObject["config"] as? [String: Any])
        #expect(onConfig["text_polishing"] == nil)
        #expect(onConfig["outputs"] == nil)

        let off = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.englishModel,
            language: "en", vocabulary: [], trackingId: "trk")
        let offObject = try #require(try JSONSerialization.jsonObject(with: off) as? [String: Any])
        #expect(offObject["outputs"] == nil)

        let detection = try ZoomScribeClient.prepareRequestBody(
            audioData: Data([1]), fileName: "call.wav", model: ZoomScribe.multilingualModel,
            language: nil, vocabulary: [], trackingId: "trk", textPolishing: true)
        let detectionObject = try #require(try JSONSerialization.jsonObject(with: detection) as? [String: Any])
        #expect(detectionObject["outputs"] == nil)
    }

    @Test func fastParsesPolishedTextAndSkipsFailureMarkers() {
        let polished = Data(#"{"result":{"text_display":"hello"},"outputs":{"text_polishing":{"text":"[Speaker 1] Hello there."}}}"#.utf8)
        #expect(ZoomScribeClient.parsePolishedTranscript(from: polished) == "Hello there.")
        let failed = Data(#"{"result":{"text_display":"hi"},"outputs":{"text_polishing":{"text":"hi [polishing_failed: timeout]"}}}"#.utf8)
        #expect(ZoomScribeClient.parsePolishedTranscript(from: failed) == nil)
        #expect(ZoomScribeClient.parsePolishedTranscript(from: Data(#"{"result":{"text_display":"hi"}}"#.utf8)) == nil)
    }

    @Test func sessionUpdateAddsTopLevelOutputsOnlyWhenPolishingEnabled() throws {
        let on = ZoomScribe.sessionUpdateMessage(
            language: "en", model: ZoomScribe.englishModel, vocabulary: [], textPolishing: true)
        let onObject = try #require(try JSONSerialization.jsonObject(with: Data(on.utf8)) as? [String: Any])
        let outputs = try #require(onObject["outputs"] as? [String: Any])
        let polishing = try #require(outputs["text_polishing"] as? [String: Any])
        #expect(polishing["enabled"] as? Bool == true)
        #expect(Set(onObject.keys) == ["type", "audio", "config", "outputs"])
        let onConfig = try #require(onObject["config"] as? [String: Any])
        #expect(onConfig["text_polishing"] == nil)

        let off = ZoomScribe.sessionUpdateMessage(language: "en", model: ZoomScribe.englishModel, vocabulary: [])
        let offObject = try #require(try JSONSerialization.jsonObject(with: Data(off.utf8)) as? [String: Any])
        #expect(offObject["outputs"] == nil)
    }

    @Test func completedEventUsesPolishedTextAndFallsBackWithoutIt() {
        let polished = #"{"type":"transcription.completed","transcript":"hello [channel0] there","outputs":{"text_polishing":{"text":"[Speaker 1] Hello there."}}}"#
        #expect(ZoomScribe.parseServerEvent(polished) == .transcript("Hello there."))

        let failed = #"{"type":"transcription.completed","transcript":"hello there","outputs":{"text_polishing":{"text":"hello there [polishing_failed: partial]"}}}"#
        #expect(ZoomScribe.parseServerEvent(failed) == .transcript("hello there"))

        let plain = #"{"type":"transcription.completed","transcript":"hello there."}"#
        #expect(ZoomScribe.parseServerEvent(plain) == .transcript("hello there."))
    }

    @Test func deltasAreNeverPolished() {
        let delta = #"{"type":"transcription.delta","delta":"Hel","outputs":{"text_polishing":{"text":"Hello."}}}"#
        #expect(ZoomScribe.parseServerEvent(delta) == .delta("Hel"))
    }

    @Test func cleansesSpacesOnlyForChineseModel() {
        #expect(ZoomScribe.cleanedTranscript("合 国 推 动", model: "zoom-scribe-zh") == "合国推动")
        #expect(ZoomScribe.cleanedTranscript("土 地， 让", model: "zoom-scribe-zh") == "土地，让")
        #expect(ZoomScribe.cleanedTranscript("VIN 123 码", model: "zoom-scribe-zh") == "VIN 123码")
        #expect(ZoomScribe.cleanedTranscript("VIN 123 A", model: "zoom-scribe-zh") == "VIN 123 A")
        #expect(ZoomScribe.cleanedTranscript("合\u{3000}国", model: "zoom-scribe-zh") == "合国")

        #expect(ZoomScribe.cleanedTranscript("合 国", model: "zoom-scribe-en") == "合 国")
        #expect(ZoomScribe.cleanedTranscript("合 国", model: ZoomScribe.multilingualModel) == "合 国")
        #expect(ZoomScribe.cleanedTranscript("合 国", model: nil) == "合 国")
    }

    @Test func liveCompletedEventTextIsCleanedAfterParsing() {
        // The streaming client applies cleanup after parsing, so this checks the parsed text it cleans.
        let completed = #"{"type":"transcription.completed","transcript":"合 国 推 动"}"#
        guard case .transcript(let text) = ZoomScribe.parseServerEvent(completed) else {
            Issue.record("Expected a transcript event")
            return
        }
        #expect(ZoomScribe.cleanedTranscript(text, model: "zoom-scribe-zh") == "合国推动")
    }
}
