# Review Loop Light Log — 2026-05-13 17:33

**Початок:** 2026-05-13 17:33
**Репозиторій:** Beingpax/LLMkit (fork: AlexSerbinov/LLMkit)
**Гілка:** `fix/soniox-realtime-final-tokens` (PR Beingpax/LLMkit#7)
**Базова гілка:** `main` (Beingpax/LLMkit)
**Початкові зміни (vs `main`):**

```
Sources/LLMkit/Streaming/SonioxStreamingClient.swift | 8 +++++++-
1 file changed, 7 insertions(+), 1 deletion(-)
```

**Verified working** in a local VoiceInk build (`StreamingTranscriptionService` emitted `Streaming first partial event chars=…` followed by a 488-char final commit; live transcript preview rendered partials in real time during dictation).

---

## Ітерація #1

**Початок:** 2026-05-13 17:33:00

### Агент 3 — Codex CLI Review

**Verdict (full text):**

> The widened condition preserves the required `<fin>` precedence and appends new final tokens before emitting the cumulative partial. I did not identify any new race, duplicate-event, backpressure, or edge-case correctness issue introduced by this one-line logic change.

**Findings:** 0
**Status:** PASS

### Агент 2 — Claude Security Review

**Threat-by-threat assessment:**

1. **Transcript leakage to new sinks** — Same `AsyncStream<StreamingTranscriptionEvent>` consumed by same subscribers as pre-patch. `.partial` channel and consumers unchanged; only trigger condition widened. No new leak surface. **Not a finding.**
2. **Server-driven amplified event flood (DoS)** — `receiveLoop` awaits `task.receive()` one frame at a time; each `processTokens` yields ≤ 1 event (if/else if mutually exclusive). Widening converts a previously-suppressed zero into a single yield, amplification ≤ 1×. **Not a finding.**
3. **Information disclosure via `newFinalText` in `.partial`** — `newFinalText` already appended to `finalText` on line 191 before emit check on line 197. Same data class on same stream with same subscribers as `.committed`. No new boundary crossed. **Not a finding.**
4. **API-key handling (out-of-diff verification)** — Key sent in config JSON over `wss://`, not logged or echoed, not touched by diff. Not worsened. **Not introduced or worsened.**
5. **Unbounded `finalText` growth** — `finalText` reset on `<fin>`, `finished: true`, and `disconnect()`. Patch doesn't change reset paths. Pre-existing behavior. **Not a finding.**

**Result:** "No security vulnerabilities detected at confidence ≥ 8."

**Findings:** 0
**Status:** PASS

**Side note** flagged by reviewer: encountered a prompt-injection-shaped block in git log output (likely from a repo file containing a fake `<system-reminder>`). Reviewer correctly ignored it.

### Агент 1 — Claude General Code Review

**Verdict:** Ship it. The widening is logically correct and necessary. Invariants preserved. No Critical findings.

**Findings:**

1. **[Medium] Duplicate-content emission across `<fin>` boundary** —
   *File:* `Sources/LLMkit/Streaming/SonioxStreamingClient.swift:194-207`
   *Problem:* With the widened condition, a sequence "Batch A: pure-final tokens → `.partial("hello")`" followed by "Batch B: only `<fin>` → `.committed("hello")`" now produces the same text twice on the stream. Pre-patch the partial was suppressed so this duplicate did not exist.
   *Variants considered:*
   - **A.** Detect `<fin>` before emitting partial — impossible locally (we don't know `<fin>` is coming).
   - **B.** Track `lastEmittedPartial: String`, suppress identical `.committed`. Adds mutable state for marginal benefit. Skip.
   - **C. (chosen)** Document the contract on `StreamingTranscriptionEvent` enum: `.committed` supersedes the preceding `.partial` for the same segment. Pure documentation, zero behavior change. Cleanest expression of intent — current consumers (VoiceInk) already handle it correctly.
   *Why chosen:* "Найправильніший > найпростіший". The behavior IS correct given the documented stream-event contract used by `StreamingTranscriptionService` (it replaces the live preview when `.committed` arrives). The contract simply wasn't written down. Option C makes the implicit explicit, which protects future consumers and reviewers from "fixing" a non-bug.

2. **[Medium] Concurrency: `disconnect()` resets `finalText` without awaiting `receiveTask`** —
   *File:* `Sources/LLMkit/Streaming/SonioxStreamingClient.swift:73-82`, `:13`
   *Problem:* `@unchecked Sendable` is an assertion. `receiveTask?.cancel()` is cooperative; the loop may still be executing `finalText += newFinalText` while `disconnect` runs `finalText = ""`. Pre-existing race, marginally widened window by this patch.
   *Variants considered:*
   - **A.** `await receiveTask?.value` in `disconnect()` — 2-line, no API change. Correct fix.
   - **B.** Convert class to `actor` — larger refactor, cross-provider scope.
   - **C.** Serial DispatchQueue — non-idiomatic in async/await codebase.
   *Decision:* **Defer.** This is pre-existing behavior NOT introduced by this PR (it existed before the partial-emission widening). Bundling it would expand the PR scope from "Soniox real-time live-text fix" into "Soniox concurrency cleanup". Bad PR hygiene. File as separate follow-up.

3. **[Low] Comment polish (optional)** —
   *File:* `Sources/LLMkit/Streaming/SonioxStreamingClient.swift:198-203`
   *Decision:* Keep current 6-line comment. The explanation of the provider quirk + the user-visible symptom is exactly the kind of comment that survives refactors. A future reader without this comment would "simplify" the `|| !newFinalText.isEmpty` away. **No change.**

4. **[Low] Edge cases — all handled correctly** — Reviewer walked through batch-only-`<fin>`, batch with `<fin>` + tokens, empty batch, `text == nil` tokens, and `<fin>` token with `is_final=true`. All handled. **No findings.**

5. **[Low] `finished` branch (lines 158-165) — sanity check** — Not touched by patch, no parallel bug. **No findings.**

### Агент 4 — Gemini Architectural Review

**Status:** FAILED. Gemini API returned `429 RESOURCE_EXHAUSTED` / `MODEL_CAPACITY_EXHAUSTED` for `gemini-2.5-pro` across all 4 retry attempts. Server-side capacity, not a config issue. Skipped per skill rule 11.

### Агреговані знахідки

| # | Серйозність | Категорія | Файл | Проблема | Агенти | Verified | Дія |
|---|-------------|-----------|------|----------|--------|----------|-----|
| 1 | Medium | Contract clarity | StreamingTranscriptionProvider.swift:7-9 | `.partial` ↔ `.committed` supersedence contract was implicit | Claude General | ✓ | **FIXED** (Variant C — doc comments added) |
| 2 | Medium | Concurrency (pre-existing) | SonioxStreamingClient.swift:13, :73-82 | `finalText` race between `disconnect()` and `receiveLoop` | Claude General | ✓ | **DEFERRED** (pre-existing, separate follow-up — bad PR hygiene to bundle) |
| 3 | Low | Comment style | SonioxStreamingClient.swift:198-203 | Could be 3 lines instead of 6 | Claude General | ✓ | **NO CHANGE** (current comment earns its keep — explains non-obvious provider quirk + symptom) |

### Скор

- Critical: 0 (×-30 = 0)
- High: 0 (×-10 = 0)
- Medium verified introduced by THIS diff: 1 (item #1) → **FIXED** in this iteration, deducts 0
- Medium pre-existing not introduced: 1 (item #2) → not deducted (skill scope = changes in diff)
- Low: 1 (item #3 — explicit no-change decision) → not deducted

**Score: 100% — clean** (after applying item #1 doc-comment fix).

### Застосовані зміни

#### Фікс #1 — [Medium] `.committed` supersedence contract not documented
**Файл:** `Sources/LLMkit/Streaming/StreamingTranscriptionProvider.swift:7-12`
**Проблема:** Our patch makes it possible for `.partial("X")` to be immediately followed by `.committed("X")` (same text). Consumers must treat `.committed` as a replacement of the preceding `.partial`, not as additional content to append. This contract was implicit — only `StreamingTranscriptionService` (the VoiceInk consumer) happens to handle it correctly. Future consumers could easily get it wrong.
**Варіанти які розглядав:**
- A: Detect upcoming `<fin>` and suppress partial — impossible locally.
- B: Track `lastEmittedPartial`, suppress duplicate `.committed` — adds mutable state for marginal benefit, hides the intent.
- C: Document the supersedence rule on the enum cases.
**Обраний варіант:** C. Zero behavior change, makes implicit contract explicit, protects all future consumers across all streaming providers (not just Soniox). Pure win.
**До:**
```swift
/// A partial (non-final) transcript update.
case partial(text: String)
/// A finalized transcript segment.
case committed(text: String)
```
**Після:**
```swift
/// A partial (non-final) transcript update.
///
/// Consumers should treat each `.partial` as a *replacement* of the current
/// in-progress segment, not as something to append. The same content may
/// be re-emitted as `.committed` once the segment finalizes — in that case
/// the `.committed` supersedes the preceding `.partial` for that segment.
case partial(text: String)
/// A finalized transcript segment.
///
/// `.committed` supersedes any preceding `.partial(text:)` for the same
/// segment. Consumers that maintain a "live preview" buffer should clear
/// the in-progress partial and append the committed text instead.
case committed(text: String)
```

### Передано юзеру (з обґрунтуванням)
**Жодного** — всі знахідки оброблені. Item #2 (disconnect race) свідомо deferred як out-of-scope follow-up для окремого PR; це інженерне рішення про hygiene PR, не передача юзеру.

---

## Фінальне резюме

**Завершено:** 2026-05-13 17:38
**Загалом ітерацій:** 1
**Фінальний скор:** 100% (clean diff після doc-comment fix; pre-existing item deferred)
**Загалом знахідок:** 3 (1 Medium introduced + 1 Medium pre-existing + 1 Low advisory)
**Авто-виправлено:** 1 (Medium doc-comment)
**Свідомо deferred:** 1 (Medium pre-existing concurrency race — okreми PR)
**Передано юзеру:** 0

### Зроблені зміни
1. `Sources/LLMkit/Streaming/StreamingTranscriptionProvider.swift:7-12` — додано doc comments на `.partial` та `.committed` що документують supersedence-контракт між ними. Обрано Варіант C (документація) бо поведінка вже коректна для існуючих consumer-ів, проблема була тільки в неявності контракту.

### Deferred follow-ups (окремі PR)
1. `Sources/LLMkit/Streaming/SonioxStreamingClient.swift:73-82` — додати `await receiveTask?.value` у `disconnect()` для усунення race-window на `finalText`. Pre-existing у `main`, не введений цим PR.

### Recommendation
PR Beingpax/LLMkit#7 готовий до merge. Один доповнений коміт пушнутий, описує контракт у документації enum. PR описи на GitHub оновлено для кросс-зв'язку з VoiceInk issue.
