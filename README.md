# LLMkit

A lightweight Swift package that provides stateless API clients for cloud transcription, LLM chat completion, and real-time streaming transcription services. Built for [VoiceInk](https://github.com/anthropics/VoiceInk), but usable in any Swift project.

## What It Does

LLMkit is a pure networking layer — no UI, no persistence, no app state. It handles HTTP requests, multipart uploads, WebSocket connections, error mapping, and automatic retries with exponential backoff on transient failures (429, 5xx).

## Clients

**Transcription** (audio file → text):
- `DeepgramClient`, `ElevenLabsClient`, `SonioxClient`, `MistralTranscriptionClient`, `GeminiTranscriptionClient`, `OpenAITranscriptionClient`

**LLM Chat Completion** (messages → response):
- `AnthropicLLMClient`, `OpenAILLMClient`, `OllamaClient`, `OpenRouterClient`

**Streaming Transcription** (live audio → real-time text via WebSocket):
- `ElevenLabsStreamingClient`, `DeepgramStreamingClient`, `MistralStreamingClient`, `SonioxStreamingClient`

## Usage

All batch clients use static methods — no initialization needed:

```swift
import LLMkit

let text = try await DeepgramClient.transcribe(
    apiKey: "...", audioData: data, model: "nova-3", language: "en"
)

let response = try await AnthropicLLMClient.chatCompletion(
    apiKey: "...", model: "claude-sonnet-4-5-20250929", messages: [.user("Hello")]
)
```

Streaming clients are class-based with `AsyncStream` event delivery:

```swift
let client = DeepgramStreamingClient()
try await client.connect(apiKey: "...", model: "nova-3", language: "en")
for await event in client.transcriptionEvents { /* .partial, .committed, .error */ }
```

## Requirements

- Swift 6.2+, macOS 14+, iOS 17+
- Zero external dependencies — uses only Foundation and `URLSession`

## Error Handling

All clients throw `LLMKitError` with cases for missing keys, HTTP errors, network failures, decoding issues, and timeouts.
