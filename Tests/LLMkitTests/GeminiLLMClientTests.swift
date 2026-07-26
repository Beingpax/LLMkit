import Foundation
import Testing
@testable import LLMkit

@Suite
struct GeminiLLMClientTests {
    @Test
    func testBuildsStableNativeInteractionRequest() throws {
        let request = try GeminiLLMClient.makeChatCompletionRequest(
            apiKey: "test-key",
            model: "gemini-3.6-flash",
            messages: [
                .system("Ignored system message"),
                .user("First question"),
                .assistant("First answer"),
                .user("Second question"),
            ],
            systemPrompt: "Explicit instruction",
            thinkingLevel: .minimal,
            store: false
        )

        #expect(
            request.url?.absoluteString
                == "https://generativelanguage.googleapis.com/v1/interactions"
        )
        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "x-goog-api-key") == "test-key")
        #expect(request.value(forHTTPHeaderField: "Authorization") == nil)

        let body = try #require(request.httpBody)
        let json = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )

        #expect(json["model"] as? String == "gemini-3.6-flash")
        #expect(json["system_instruction"] as? String == "Explicit instruction")
        #expect(json["store"] as? Bool == false)
        #expect(json["temperature"] == nil)
        #expect(json["top_p"] == nil)
        #expect(json["top_k"] == nil)
        #expect(json["stream"] == nil)

        let generationConfig = try #require(json["generation_config"] as? [String: Any])
        #expect(generationConfig["thinking_level"] as? String == "minimal")

        let input = try #require(json["input"] as? [[String: Any]])
        #expect(input.count == 3)
        #expect(input[0]["type"] as? String == "user_input")
        #expect(input[1]["type"] as? String == "model_output")
        #expect(input[2]["type"] as? String == "user_input")
    }

    @Test
    func testUsesSystemMessagesWhenExplicitPromptIsMissing() throws {
        let request = try GeminiLLMClient.makeChatCompletionRequest(
            apiKey: "test-key",
            model: "gemini-3.5-flash-lite",
            messages: [
                .system("First instruction"),
                .system("Second instruction"),
                .user("Hello"),
            ],
            systemPrompt: nil,
            thinkingLevel: nil,
            store: false
        )

        let body = try #require(request.httpBody)
        let json = try #require(
            JSONSerialization.jsonObject(with: body) as? [String: Any]
        )
        #expect(
            json["system_instruction"] as? String
                == "First instruction\nSecond instruction"
        )
        #expect(json["generation_config"] == nil)
    }

    @Test
    func testRejectsPrefilledModelTurn() {
        do {
            _ = try GeminiLLMClient.makeChatCompletionRequest(
                apiKey: "test-key",
                model: "gemini-3.6-flash",
                messages: [.user("Hello"), .assistant("Prefill")],
                systemPrompt: nil,
                thinkingLevel: .minimal,
                store: false
            )
            Issue.record("Expected encodingError")
        } catch LLMKitError.encodingError {
            // Expected.
        } catch {
            Issue.record("Expected encodingError, received \(error)")
        }
    }

    @Test
    func testDecodesLastConsecutiveTextBlocks() throws {
        let data = Data(
            """
            {
              "steps": [
                {"type": "thought"},
                {
                  "type": "model_output",
                  "content": [
                    {"type": "text", "text": "Earlier"},
                    {"type": "image", "uri": "example"},
                    {"type": "text", "text": "Final "},
                    {"type": "text", "text": "answer"}
                  ]
                }
              ]
            }
            """.utf8
        )

        #expect(
            try GeminiLLMClient.decodeChatCompletionResponse(from: data)
                == "Final answer"
        )
    }

    @Test
    func testThrowsWhenResponseHasNoTextOutput() {
        let data = Data(
            """
            {"steps": [{"type": "thought"}]}
            """.utf8
        )

        do {
            _ = try GeminiLLMClient.decodeChatCompletionResponse(from: data)
            Issue.record("Expected noResultReturned")
        } catch LLMKitError.noResultReturned {
            // Expected.
        } catch {
            Issue.record("Expected noResultReturned, received \(error)")
        }
    }
}
