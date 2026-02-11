import Foundation

/// A chat message used in LLM completions (OpenAI-compatible format).
public struct ChatMessage: Codable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }

    /// Convenience factory for a system message.
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: "system", content: content)
    }

    /// Convenience factory for a user message.
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: "user", content: content)
    }

    /// Convenience factory for an assistant message.
    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: "assistant", content: content)
    }
}
