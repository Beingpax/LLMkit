import Foundation
import Testing

@testable import LLMkit

struct OllamaClientTests {
    @Test func samplingOptionsAreNestedUnderOptions() throws {
        let json = try requestJSON(
            options: OllamaGenerationOptions(
                temperature: 1.0,
                topP: 0.95,
                topK: 64
            )
        )
        let options = try #require(json["options"] as? [String: Any])

        #expect(json["temperature"] == nil)
        #expect(json["top_p"] == nil)
        #expect(json["top_k"] == nil)
        #expect(options["temperature"] as? Double == 1.0)
        #expect(options["top_p"] as? Double == 0.95)
        #expect(options["top_k"] as? Int == 64)
    }

    @Test func nilOptionsUseModelDefaults() throws {
        let json = try requestJSON(options: nil)

        #expect(json["options"] == nil)
        #expect(json["temperature"] == nil)
    }

    @Test func emptyOptionsUseModelDefaults() throws {
        let json = try requestJSON(options: OllamaGenerationOptions())

        #expect(json["options"] == nil)
    }

    @Test func partialOptionsOmitUnsetValues() throws {
        let json = try requestJSON(options: OllamaGenerationOptions(topK: 64))
        let options = try #require(json["options"] as? [String: Any])

        #expect(options["temperature"] == nil)
        #expect(options["top_p"] == nil)
        #expect(options["top_k"] as? Int == 64)
    }

    private func requestJSON(options: OllamaGenerationOptions?) throws -> [String: Any] {
        let request = try OllamaClient.makeGenerateRequest(
            baseURL: URL(string: "http://localhost:11434")!,
            model: "test-model",
            prompt: "user prompt",
            systemPrompt: "system prompt",
            options: options,
            think: false
        )
        let body = try #require(request.httpBody)
        return try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
    }
}
