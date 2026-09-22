import Foundation
import Testing

@testable import LLMkit

struct GeminiStreamingClientTests {
    @Test func defaultInitializerUsesVerbatim() throws {
        let message = try GeminiStreamingClient().makeSetupMessage(
            model: "gemini-3.5-transcribe", language: nil, customVocabulary: []
        )
        let data = try JSONEncoder().encode(message)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let setup = try #require(json["setup"] as? [String: Any])
        let transcription = try #require(setup["inputAudioTranscription"] as? [String: Any])
        #expect(transcription["mode"] as? String == "VERBATIM")
    }

    @Test(arguments: [GeminiTranscriptionMode.verbatim, .smart])
    func setupEncodesLiveModeAndPreservesRecognitionHints(mode: GeminiTranscriptionMode) throws {
        let setup = try setupJSON(mode: mode, language: "en-US", vocabulary: [" VoiceInk ", "voiceink", "", "LLMkit"])
        #expect(setup["model"] as? String == "models/gemini-3.5-transcribe-live")
        let generation = try #require(setup["generationConfig"] as? [String: Any])
        #expect(generation["responseModalities"] as? [String] == ["TEXT"])
        let transcription = try #require(setup["inputAudioTranscription"] as? [String: Any])
        #expect(transcription["mode"] as? String == (mode == .smart ? "SMART" : "VERBATIM"))
        #expect(transcription["languageCodes"] as? [String] == ["en-US"])
        #expect(transcription["customVocabulary"] as? [String] == ["VoiceInk", "LLMkit"])
    }

    @Test(arguments: [nil, "", "auto"] as [String?])
    func smartModePreservesAutomaticLanguageDetection(language: String?) throws {
        let setup = try setupJSON(mode: .smart, language: language, vocabulary: [], model: "gemini-3.5-transcribe-live")
        let transcription = try #require(setup["inputAudioTranscription"] as? [String: Any])
        #expect(transcription["languageCodes"] as? [String] == [])
        #expect(transcription["mode"] as? String == "SMART")
    }

    @Test func rejectsUnsupportedModels() {
        #expect(throws: LLMKitError.self) {
            try GeminiStreamingClient(mode: .smart).makeSetupMessage(
                model: "unsupported", language: nil, customVocabulary: []
            )
        }
    }

    private func setupJSON(
        mode: GeminiTranscriptionMode,
        language: String?,
        vocabulary: [String],
        model: String = "gemini-3.5-transcribe"
    ) throws -> [String: Any] {
        let message = try GeminiStreamingClient(mode: mode).makeSetupMessage(
            model: model, language: language, customVocabulary: vocabulary
        )
        let data = try JSONEncoder().encode(message)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try #require(json["setup"] as? [String: Any])
    }
}
