# Use cases — what to run, what to expect, what trips people up

A practical guide matched to common workloads. For each: which compose to boot, why it fits, what gotchas to know about, and what limitations to plan around.

> If you're new to the stack, read the [README](../README.md) first. This doc assumes you've got it running and want to dial it in for your specific workflow.

---

## Quick map

| Your workload | Boot this | Expected ctx | Expected TPS |
|---|---|---|---|
| General chat / quick Q&A (≤20K) | `docker-compose.fast-chat.yml` | 20K | 55 narr / 70 code |
| Tool-using agents (Cline, Roo, Cursor, Hermes, OpenAI Assistants) | `docker-compose.yml` (default) | 48K | 51 narr / 68 code |
| Coding (single-file or repo-aware with tool truncation) | `docker-compose.yml` (default) | 48K | 51 narr / 68 code |
| Long single prompts (RAG, document summarization) | `docker-compose.tools-text.yml` | 75K, no vision | 53 narr / 70 code |
| Vision (images in prompts) | `docker-compose.yml` (default) | 48K | 51 narr / 68 code |
| Whole-codebase / long-doc agents (need 128K+) | default + opt-in 128K-205K | up to 205K | 50-51 narr / 66-68 code |
| Simplest stack / no patches | `docker-compose.minimal.yml` | 32K | 32 / 33 |
| Skip Genesis but keep MTP TPS | `docker-compose.no-genesis-mtp.yml` | 20K | 55 / 68 |

---

## General chat / Q&A

**Best for:** ChatGPT-replacement workloads. Short user messages, short-to-medium replies, occasional long-form essays. Browser-based UIs (Open WebUI, LibreChat).

**Recommended:** [`docker-compose.fast-chat.yml`](../compose/docker-compose.fast-chat.yml) (20K + fp8 KV). Or the default if you also use it for agent work and don't want to switch composes.

**Why fast-chat over default:** ~5-7% faster TPS at small context; fp8 KV has a smaller activation footprint than TQ3 KV. If 20K is enough for your conversation depth, this is the cleanest path.

**Gotchas:**
- 20K context fills up faster than you'd think with system prompts + tool definitions + history. A 30-turn coding conversation can exceed 20K. If users will let chats grow long, switch to the default (48K) instead.
- Vision works at this ctx but each image consumes tokens (~640-1280 per image at standard resolution); image-heavy chats burn through 20K fast.

**Limitations:**
- No way to push past 20K on this compose without switching configs. Request larger context = HTTP 400 from the engine pre-check.

**Tuning levers (rare for chat):**
- `temperature=0.6, top_p=0.95, top_k=20` are Qwen3 defaults; lower temp for more deterministic output.
- For a "concise mode," set `chat_template_kwargs.enable_thinking=false` (default) so the model skips the reasoning preamble. For "show your work" mode, set it to `true`.

---

## Tool-using agents (Cline, Roo, Cursor, Hermes, OpenAI Assistants)

**Best for:** AI coding assistants that call tools (read_file, run_command, web_fetch). Multi-turn conversations with structured tool returns.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — 48K + TQ3 KV).

**Why default over fast-chat:** Tool returns can be sizable (a `read_file` on a 500-line source file is ~5-15K tokens). Combined with growing conversation history, 20K runs out fast. 48K covers most realistic agent loops with comfortable headroom.

**Gotchas:**
- **Tool response size matters.** A single tool that returns 50K+ tokens (e.g., a `web_fetch` on a long article, or `read_file` on a giant log) will OOM the engine. **This is the single most common production crash.** Most agent frameworks (Cline, OpenAI Assistants, LangChain) have a default tool-output truncation around 10-20K tokens; **verify your framework's truncation is enabled and tuned**.
- **Tool calling needs Genesis v7.14.** The default compose loads it; check container logs for `Genesis Results: 27 applied` to confirm. Without it, the model emits `<tool_call>` as plain text instead of populating `tool_calls[]` cleanly (the silent cascade bug).
- **Streaming with tool calls** works; SSE chunks include the `tool_calls` delta as the model emits them. Test with `verify-full.sh #5` (streaming) + `#4` (tool calls).

**Limitations:**
- No inline image generation (the model can't make images, only see them with vision).
- Single-prompt cap is ~50K tokens at this config — if your agent stuffs back a single 50K+ tool response, it'll OOM. Keep tool truncation on.
- 48K total ctx caps how long an agent loop can run before context recycling. For very long-running agents, opt into 96K-128K (see Frontier Context below).

**Tuning levers:**
- For more deterministic tool-call generation, lower temperature to `0.3` and set `tool_choice="auto"` or specific.
- `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are baked into the default compose — don't change unless you know what they do.

---

## Coding (single-file or repo-aware)

**Best for:** Code generation, code review, refactoring, debugging. With or without tool access.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — 48K).

**Why default:** Coding workflows often paste in 1-3 source files (3-15K tokens each), then iterate. 48K handles ~3-5 files at typical sizes plus reasoning history. Code TPS at 67-68 is the second-fastest config (only fast-chat at 70 beats it).

**Gotchas:**
- **Code TPS (67-68) > narrative TPS (51).** This is normal — the MTP draft model has higher acceptance rate on structured tokens (function names, brackets, keywords). Don't be surprised by the asymmetry.
- The model is a **generalist** — it codes well but isn't specialized. For pure-code workloads on this hardware tier, dedicated coder models (DeepSeek-Coder, Qwen2.5-Coder) may match TPS at smaller VRAM.
- **Repository-scale context (5+ files, dependency graphs, etc.):** 48K may not be enough. Consider opting into 96K-128K. Be aware of the prefill cliff at 50-60K single-prompt tokens.

**Limitations:**
- No streaming-tool-call hot-loop optimization — each tool round trip is a full prefill of accumulated history.
- The model is good at Python/TypeScript/Rust/Go; less reliable on niche languages (Erlang, Forth, etc.) — verify on small examples before trusting.

**Tuning levers:**
- For deterministic code: `temperature=0.0` (greedy). Sacrifices some creativity but reproducible.
- For creative refactor: `temperature=0.7-0.9`.
- `enable_thinking=true` for tricky algorithm work — the model thinks through the approach before emitting code.

---

## Long single prompts (RAG, document summarization)

**Best for:** Loading a long document or document-set in one shot, asking questions about it. Single-shot summarization. Not multi-turn agent loops.

**Recommended:** [`docker-compose.tools-text.yml`](../compose/docker-compose.tools-text.yml) (75K + fp8 KV, no vision).

**Why tools-text over the default:** fp8 KV avoids the GDN-forward prefill cliff at 50-60K-token single prompts. We tested up to 60K-token single-prompt depth on this config and it works. The default's TQ3 KV pathway hits the cliff there.

**Gotchas:**
- **No vision.** The compose ships with `--language-model-only`. If you need images alongside the long doc, you'll need to switch composes (and accept lower max-prompt size on the default).
- **First request is slow.** A 60K-token cold prefill takes 30-90 seconds at typical TQ3 prefill rates (faster on fp8 — closer to 30s). Subsequent requests against the same prefix hit the prefix cache and feel instant.
- **75K is the engine ceiling**, not the safe-prompt ceiling. Stay under ~60K total prompt to leave room for the response and any incremental history.

**Limitations:**
- No vision (see gotcha).
- Spec-decode acceptance (AL ~3.4) drops modestly at very long context as the draft model's predictions get less aligned. Still net-positive over greedy.

**Tuning levers:**
- For pure summarization: `temperature=0.3` and a system prompt directing the model to summarize without commentary.
- For Q&A with citations: `enable_thinking=true` to get the model to reason about which parts of the doc are relevant before answering.
- Pre-warm the prefix cache before user-facing use: send a dummy request with the document loaded; subsequent real requests will skip prefill.

---

## Vision (images in prompts)

**Best for:** Multimodal chat where users paste screenshots, photos, diagrams. Code review on UI mockups, document OCR-style tasks, "what's in this picture" Q&A.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) (default — 48K with vision tower active). Or `fast-chat.yml` if you only need ≤20K total ctx.

**Why default:** Vision tower is on. The `MoonViT` BF16 implementation is small (~1 GB VRAM) but adds activation overhead.

**Gotchas:**
- **Each image consumes tokens.** Default resolution images are ~640-1280 tokens depending on aspect ratio. A 5-image conversation can chew through several thousand tokens before any text gets processed. Plan ctx budget accordingly.
- **Vision quality** is good for charts, screenshots, and natural images. Less reliable for OCR on dense text — Qwen team didn't optimize this model for OCR specifically.
- **Image input format:** OpenAI-compatible — pass as `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` or `{"url": "https://..."}`. Both work.
- **No image generation** — this model is vision-input-only. For "draw me a picture" use a separate diffusion model (ComfyUI / Stable Diffusion / etc.).

**Limitations:**
- Drop vision if you need 192K+ context (`--language-model-only` flag in v714.yml frees ~1 GB and lifts the engine ceiling from 192K to ~206K).
- High-resolution images (2048×2048+) are downsampled internally; details below the model's vision-tower resolution won't be processed.

**Tuning levers:**
- For code-screenshot work: `enable_thinking=true` so the model reasons about layout before describing.
- If you don't need vision in a particular request, skip the image content blocks; the vision tower stays loaded but won't do work.

---

## Frontier context (128K-205K) — whole-codebase / long-document agents

**Best for:** Running an agent over an entire codebase, processing very long documents in one shot, research workflows on book-length texts.

**Recommended:** [`docker-compose.yml`](../compose/docker-compose.yml) with **opt-in tier change**. Edit the file:

```yaml
# Change these two lines:
- --max-model-len
- "128000"          # or 192000, or 205000
- --gpu-memory-utilization
- "0.95"            # 0.95 for 128K, 0.98 for 192K/205K
```

For 205K, also uncomment `--language-model-only` (drops vision, frees ~1 GB).

**Why this is opt-in not default:** The default ships at 48K because both prefill cliffs (TurboQuant tool-prefill at 25K+ and DeltaNet GDN forward at 50-60K) sit below the higher tiers. Pushing context up reduces activation headroom and exposes you to OOM crashes on big single prompts. **Only pick higher tiers if you understand your workload and can keep individual prompts within the matching safe envelope.**

**Gotchas:**
- **Opt-in tiers are NOT safe under unrestricted tool prefills.** A single tool returning 25K+ tokens at 192K-config will OOM the engine (this is ampersandru's [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) bug class — kept as documentation, not regression).
- **Single-prompt safety is roughly the same across tiers.** Tier choice affects **how big your KV pool is** (i.e., conversation history capacity), not how big a single prompt can be. The GDN cliff at ~50-60K applies regardless of `--max-model-len`.
- **Cold prefill at 192K is genuinely slow** — 3-5 minutes for a fresh 150K-token doc. Use prefix caching aggressively.

**Limitations:**
- Recall quality at 100K+ tokens degrades — like most current LLMs, attention thins toward the document middle. Test recall on your actual doc/code corpus before trusting.
- Throughput drops slightly at frontier ctx (50-51 narr at 192K vs 51 narr at 48K). Modest but measurable.

**Tuning levers:**
- For very long documents: process them in chunks of 30-40K each rather than one 150K request — better recall, no OOM risk.
- For multi-file repo agents: have your agent framework summarize older turns into a compressed history (LangChain's `ConversationSummaryMemory` pattern).
- If you're consistently hitting prefill OOM, drop to a lower tier — the model performs identically, the only loss is conversation-history depth.

---

## Advanced mode — tinkering, debugging, contributing

**Best for:** Folks who want to understand the stack, reproduce results, run experiments, or contribute upstream.

**Recommended path:**

1. Boot the default. Confirm `verify-full.sh` shows all 10 checks green.
2. Run `bench.sh` and verify your numbers match the README (within ~5% run-to-run variance).
3. Read [docs/INTERNALS.md](INTERNALS.md) for the engineering story (3 hurdles, 9-probe forensics, upstream tracking).
4. Read [CHANGELOG.md](../CHANGELOG.md) for what's changed and why.
5. If contributing: open issues for repro questions, PRs for fixes. Cross-rig data is especially welcome.

**Things you might want to try:**

- **Genesis opt-in patches** — there are several env-gated patches in v7.14 beyond P64/P65/P66 (P68 long-ctx tool reminder, P69 long-ctx tool format, etc.). See `patches/genesis/vllm/_genesis/wiring/` after `setup.sh` clones it. Each is documented in its own file.
- **Different KV-cache types** — `fp8_e5m2` (default for fast-chat), `turboquant_4bit_nc`, `turboquant_k8v4`, `turboquant_3bit_nc` (default). Trade-off: smaller KV bytes per token = bigger context, but more dequant scratch + activation pressure. Configuration notes in README.
- **MTP draft length** (`num_speculative_tokens`) — n=3 is the empirical sweet spot. n=4 nominally hits higher TPS on code but 4th-position acceptance collapses to ~21%. Don't push higher.
- **Power cap** — production at 230W is quiet/cool/stable. Set to 330W via `nvidia-smi -pl 330 -i 0` for ~+10% mean TPS during heavy sessions. Past 330W: diminishing returns (SM clocks saturate near 1.9 GHz on 3090s).

**Things that won't work:**

- **GGUF on vLLM** for Qwen3-Next family — not supported upstream yet (4 PRs open + a missing `Qwen35TensorProcessor` port). Use llama.cpp / Ollama for GGUF.
- **TP=2 on a single card** — obviously. If you have 2× 3090, see [qwen36-dual-3090](https://github.com/noonghunna/qwen36-dual-3090).
- **Older drivers (< 580.x)** — vLLM nightly needs CUDA 13 runtime which needs the new driver line. Driver upgrades on stable Ubuntu LTS are non-trivial; budget time if you're behind.

---

## When something is wrong

- **`docker compose up -d` boots but `verify-full.sh` shows red checks** — most likely Genesis or `tolist_cudagraph` patch didn't apply. Check container logs for `Genesis Results: N applied` and `[tolist_cudagraph_fix] Site A: ok`. See [Troubleshooting](../README.md#troubleshooting).
- **TPS lower than expected** — re-run `bench.sh` with 3 warmups + 5 measured runs. Run-to-run variance can be 5%. If consistently low, see the README troubleshooting section.
- **OOM mid-request** — you've hit one of the two prefill cliffs. See the [Activation-memory caveat](../README.md#activation-memory-caveat-read-this-before-raising---max-model-len) in README. Either reduce single-prompt size or switch to a lower tier.
- **OOM at boot** — your `--gpu-memory-utilization` is too high for the `--max-model-len`. Lower one or both. The compose's header comment block has the working envelope per tier.

If none of the above match, open an issue with your `docker logs vllm-qwen36-27b 2>&1 | tail -200` output.
