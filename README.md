# Qwen3.6-27B on a single RTX 3090

**A validated recipe for serving Qwen3.6-27B on a single consumer 24 GB RTX 3090** — full OpenAI API, vision, tool calling, streaming, speculative decoding, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via vLLM with MTP speculative decoding + fp8_e5m2 KV cache. Built on [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) + a CUDA graph capture fix that ships in this repo.

> 📖 **Write-up:** *[Qwen3.6-27B on a single RTX 3090 — the recipe](https://medium.com/)*
> 🤝 **Companion repo:** [qwen36-dual-3090](https://github.com/noonghunna/qwen36-dual-3090) — same model on 2× 3090 (TP=2 with vision + 4-stream concurrency at 262K).
> 🐛 **Upstream bug reports:** [vllm-project/vllm#40807](https://github.com/vllm-project/vllm/issues/40807) (CUDA graph crash — worked around locally) · [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831) (TurboQuant × spec-decode output corruption — closed for ngram path via @Sandermage's v7.13 + [#40875](https://github.com/vllm-project/vllm/issues/40875) `prompt_lookup_min=8` config) · [vllm-project/vllm#40880](https://github.com/vllm-project/vllm/issues/40880) (MTP × TurboQuant × cudagraph — **root-caused by @Sandermage 2026-04-25, fixed in [Genesis v7.14](https://github.com/Sandermage/genesis-vllm-patches) via P65 cudagraph downgrade for spec-decode**; our `cudagraph_mode=NONE` workaround in `docker-compose.longctx-experimental.yml` still works at -60% TPS, but v7.14 is now the recommended fix)

---

## Status at a glance

Six configurations + five v7.14 opt-in tiers, all measured end-to-end on a single 3090 PCIe / 230W cap with bench prompts (1000-token narrative essay + 800-token quicksort code), `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (vLLM `dev205+g07351e088`, Sandermage's reference target) + Genesis v7.54. **`v714` ships 48K + 0.92 by default** — passes all 10 verify-full.sh checks including tool-response prefill (#8) and long-context needle (#7). Pick by workload; if you push past 48K read the **Activation-memory caveat** below first.

| Variant | Context | Narr TPS | Code TPS | Vision | Tools | Patches | VRAM | Notes |
|---|---|---|---|---|---|---|---|---|
| **v7.14** (`docker-compose.v714.yml`) — TQ + Genesis P65, **48K + 0.92** ⭐⭐ | **48K** | **50.9** | **67.5** | ✅ | ✅ | Genesis v7.14+ | 21.0 GB | **Production-safe default.** Below BOTH cliffs: GDN-forward (~50-60K single-prompt OOM) and TurboQuant tool prefill (~25K at high mem_util). All 10 verify-full.sh checks PASS. Vision + tools + MTP work. |
| **v7.14 — opt-in 64K** (edit v714.yml: `max-model-len=64000`) | **64K** | 50.9 | 67.5 | ✅ | ✅ | Genesis v7.14+ | 21.5 GB | More chat-history room. Single prompts >~50K may OOM in GDN forward. Tool prefills ≤40K safe. |
| **v7.14 — opt-in 96K + 0.93** | **96K** | 50.9 | 67.5 | ✅ | ✅ | Genesis v7.14+ | 22.0 GB | Useful for "long history, small individual prompts." Single prompts >~50K OOM. Tool prefills ≤30K. |
| **v7.14 — opt-in 128K + 0.95** | **128K** | 50.9 | 67.5 | ✅ | ✅ | Genesis v7.14+ | 22.3 GB | Matches GPT-4 Turbo / DeepSeek-R1 ctx tier on paper. Single prompts >~50K OOM. Tool prefills ≤40K. |
| **v7.14 — opt-in 192K + 0.98** | **192K** | 50.9 | 67.7 | ✅ | ✅ | Genesis v7.14+ | 22.3 GB | OOMs on ≥25K tool prefills ([#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) ampersandru's repro). Pick only if no big tool/doc prefills. |
| **v7.14 — opt-in 205K text-only** (also uncomment `--language-model-only`) | **205K** | 50.1 | 65.8 | ❌ | ✅ | Genesis v7.14+ | 21.5 GB | Absolute single-card ceiling on TQ3 KV (engine reports 206,400 max at 0.98). Drops vision to free ~1 GB. Same prefill-OOM caveat as 192K. |
| **Default** (`docker-compose.yml`) — fp8 KV + MTP n=3 ⭐ | 20K | **55.0** | **70.5** | ✅ | ✅ | Genesis | 22.3 GB | Best TPS at small ctx. fp8 KV sidesteps the cudagraph bug entirely. Pick when you only need ≤20K and want maximum TPS. |
| **Tools-text** (`docker-compose.tools-text.yml`) — fp8 + 75K | 75K | 53.4 | 69.6 | ❌ | ✅ | Genesis | 22.2 GB | Drops vision to free KV pool. fp8 KV — cudagraph bug doesn't fire. Faster than v7.14 at ≤75K. |
| **No-Genesis MTP** (`docker-compose.no-genesis-mtp.yml`) — fp8 + MTP, no patches | 20K | 54.7 | 68.2 | ✅ | ✅ | **none** | 22.3 GB | Identical to Default minus Genesis. Same TPS — Genesis is performance-neutral on the fp8+MTP path. Pick if you want to skip the patch tree. (Cross-rig peer: [u/sudeposutemizligi's TP=2 setup](#cross-rig-validation).) |
| **Minimal** (`docker-compose.minimal.yml`) — no spec-decode, fp8 KV | 32K | 32.4 | 32.6 | ✅ | ✅ | **none** | 20.8 GB | Simplest stack. No spec-decode → no #40880 trigger. Pure-bandwidth ceiling. |
| ~~**Long-ctx**~~ (`docker-compose.longctx-experimental.yml`) — DEPRECATED | ~~125K~~ | ~~37.9~~ | ~~49.8~~ | ✅ | ✅ | Genesis | 23.1 GB | **Superseded by v7.14.** Same effective working ceiling, lower TPS via `cudagraph_mode=NONE`. Kept for reference. |

### Activation-memory caveat (read this before raising `--max-model-len`)

Two prefill-activation cliffs make 24 GB-card defaults non-trivial. vLLM's `--gpu-memory-utilization` is a **hard cap, not a soft limit** — there's no fallback or circular buffer. Whatever's outside the cap (activation peaks, fragmentation, kernel scratch) has to fit in `(1 - mem_util) × 24 GB`. The engine's pre-check guarantees the steady-state KV fits — **not the activation peak during forward**.

**Cliff 1 — TurboQuant attention scratch + tool-response prefill** (the bug ampersandru reported in [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1)). Triggered when a ≥25K-token tool message is loaded into the conversation at high mem_util. OOM site: TurboQuant attention forward, dequant scratch + mid_o/output buffers. Allocation typically ~138 MiB.

**Cliff 2 — DeltaNet/GLA recurrent state buffer.** Triggered by any single prompt above ~50-60K tokens (regardless of tool use). OOM site: `fla.ops.chunk.chunk_gated_delta_rule_fwd_h.h.new_empty(B, NT, H, V, K)` where NT grows linearly with prompt length. Qwen3-Next is hybrid (every 4th layer is full attention; the other 3 are GDN). GDN state is sized by total seq_len — chunked-prefill doesn't help. We can't fix this without multi-GPU TP=2 or upstream `fla.ops` changes.

| Config | Engine ceiling | Safe single-prompt | Safe tool-prefill | Best for |
|---|---|---|---|---|
| **48K + 0.92** ⭐ | ~86K | up to 48K | up to 48K | **Default.** All verify-full.sh checks pass. Production-safe; engine rejects > 48K with HTTP 400. |
| 64K + 0.92 | ~86K | up to ~50K | up to 40K | Common chat + agent flows. Single prompts >50K may OOM. |
| 96K + 0.93 | ~103K | up to ~50K | up to 30K | Long history, small individual prompts. |
| 128K + 0.95 | ~140K | up to ~50K | up to 40K | Matches GPT-4-tier on paper. Same single-prompt cliff. |
| 192K + 0.98 | ~206K | up to ~16K | up to 16K | Long-ctx recall only. OOMs on big tool prefills (ampersandru-class workload). |
| 205K + 0.98 + no vision | ~206K | up to ~16K | up to 16K | Engine ceiling. Same caveats. |

#### Defense-in-depth — three places to enforce safety

1. **`--max-model-len` (vLLM, hardest)** — engine rejects requests where `input_tokens + max_tokens > limit` with **HTTP 400** before any forward pass. **This is the only true safety net** — the cleanest UX is to set max-model-len **at or below the cliff** so oversized requests fail fast with a clean error, not an OOM crash. (Why 48K is the default: just below Cliff 2's GDN onset.)

2. **Agent-framework truncation (middle layer)** — Hermes, OpenAI Assistants, Roo Cline, LangChain, Open WebUI, Cursor and similar agent frameworks **truncate tool responses** before feeding them back into the next turn. Most default to a 10-20K-token cap on individual tool outputs. This is your second line of defense — it shapes what the agent loads back, regardless of vLLM's setting. Check your framework's docs (e.g. OpenAI's `truncation_strategy`, LangChain's `length_function` callbacks) to confirm and tune.

3. **System prompts (least reliable)** — telling the model "don't fetch >X tokens" or "summarize tool returns >5K" has weak compliance — the model passes the call to the tool, the tool returns what it returns, and the system-prompt rule doesn't gate the agent's loop. Useful as a hint, not a guarantee.

In practice: **set max-model-len to your cliff (48K) for safety; let the agent framework do realistic truncation for tool responses; don't rely on system prompts as the safety mechanism.** If a user genuinely needs longer context (some research/document workflows), have them switch to a matching opt-in config and accept the trade-off explicitly.

### Decision tree (matched to use case)

- **General chat / Q&A / quick coding (≤20K)** — **Default** (`docker-compose.yml`). 55 narrative / 70 code TPS, ✅ vision, ✅ tools, fp8 KV. Best TPS; no Cliff 2 risk because 20K stays well below 50K.
- **Tool-using agents + multi-turn coding (≤48K)** — **v7.14 default** (`docker-compose.v714.yml`). 51/68 TPS with vision, full tool/streaming/thinking support, prefill-safe to 48K. **Recommended for anyone running Hermes / Cline / Roo / OpenAI Assistants on this stack.**
- **Long single prompts (single-shot 50K+ summarization or RAG)** — **Tools-text** (`docker-compose.tools-text.yml`) at 75K + fp8. fp8 KV avoids the GDN cliff at 50-60K (tested through 60K depth). Trade-off: no vision.
- **Frontier context (192K-205K) for "fits in giant book" use cases** — opt into v7.14 192K or 205K. Read the caveats first.
- **Simplest stack, no patches** — **Minimal** (`docker-compose.minimal.yml`). 32 TPS, no spec-decode, no Genesis. Zero risk.
- **Skip Genesis but keep MTP TPS** — **No-Genesis MTP** (`docker-compose.no-genesis-mtp.yml`). Same 55/68 as Default.

### Removed: eager.yml

`docker-compose.eager.yml` was originally proposed by [@ampersandru](https://github.com/ampersandru) in [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) as a 125K path that bypasses the cudagraph bug class via `--enforce-eager`. It shipped briefly with a "~52-65 TPS at 125K" claim. **Re-bench cycle on dev205 + Genesis v7.54 measured 25.5 narr / 32.3 code — strictly dominated by `longctx-experimental.yml` at the same 125K context (38/50 TPS).** `--enforce-eager` disables both cudagraph AND torch.compile, paying a real Python-overhead cost on every forward; `cudagraph_mode=NONE` keeps inductor compilation on and is faster while delivering the same context ceiling and feature set. The compose has been removed in favor of long-ctx as the recommended 125K path. Eager mode is still reachable via `--compilation-config '{"cudagraph_mode":"NONE"}' --enforce-eager` if you genuinely need the full no-graph escape hatch (e.g., for model debugging or P7 GDN dual-stream), but no shipped variant defaults to it.

### What v7.14 changes (the new variant)

[Sandermage's v7.14](https://github.com/Sandermage/genesis-vllm-patches) shipped 2026-04-25 with the P65 patch root-causing #40880: `TurboQuantAttentionImpl._prefill_attention`'s cudagraph-capture bypass treats spec-decode K+1 verify batches as first-chunk prefill (sets `cu_seqlens_k = cu_seqlens_q`), so the captured kernel ignores cached KV. Drafter and verifier both produce noise from the kernel-without-context path; for tool-call prompts they converge on the same high-bias special token (`<tool_call>`) and cascade.

P65 downgrades `_cudagraph_support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE`. vLLM's compilation auto-detects and forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch. 1-token decode batches still get piecewise capture; only K+1 spec-verify batches go eager.

This is a workaround. The proper fix is a custom multi-query Triton kernel (P67) that handles K+1 query against compressed cached KV under cudagraph capture — designed-but-not-implemented in v7.14.

### What's NOT working today

- **125K context at FULL cudagraph speed (~95 TPS) WITH tool calls** — that combination requires the proper P67 kernel. Until then, you pick two of three: long ctx, full TPS, working tools.
- **GGUF on vLLM for Qwen3-Next family** — not supported upstream yet (4 PRs open + a missing port of `Qwen35TensorProcessor` value transforms). Use llama.cpp / Ollama if you specifically need GGUF.

### Recently fixed

- **2026-04-27 (full-matrix re-bench + substrate unification)** — discovered and fixed four real compose drift bugs during a complete re-bench cycle:
  - **Image split**: composes had drifted across two different vLLM image pins (`@sha256:9bba4628a3...` = `dev21` for default/tools-text/longctx/eager; `:nightly-100c7b65...` = `dev174` for v714/minimal). All six unified to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (= `dev205+g07351e088`, Sandermage's documented reference target).
  - **`eager.yml` config drift**: shipped with `gpu-memory-utilization=0.92` and `max-model-len=131072` while [@ampersandru](https://github.com/ampersandru)'s actual measurement was `0.97` + `125000`. As-shipped failed to boot at 131K (KV-OOM). Compose deleted entirely — see "Removed: eager.yml" above.
  - **`v714.yml` mount path**: `patch_tolist_cudagraph.py` was mounted from `../patches/genesis/patch_tolist_cudagraph.py` but the file is at `../patches/patch_tolist_cudagraph.py`. Docker silently created an empty directory at the bogus path, breaking the patcher. Fixed.
  - **Bench harness regression**: commit `a381086` rewrote `scripts/bench.sh` to add streaming TTFT / CV / decode_TPS instrumentation but silently dropped the original code-prompt arm (3 narrative + 2 code → narrative only). Restored as parallel narrative + code runs in one invocation; all README TPS claims re-measured.
  - **Genesis exoneration**: ran an A/B between `default.yml` (with Genesis v7.54) and a fresh `no-genesis-mtp.yml` (identical config minus Genesis). Measured within run-to-run variance — Genesis is performance-neutral on this path, not the cause of any TPS shift vs older claims. Cross-rig confirmed by [u/sudeposutemizligi](https://www.reddit.com/r/LocalLLaMA/) on TP=2 + dev45 + no Genesis (55 narrative / 68 code, same hardware class).
- **2026-04-27** — `docker-compose.eager.yml` initial commit incorrectly claimed "no Genesis patches needed" while still using `--kv-cache-dtype turboquant_3bit_nc`. Updated to mount Genesis P4. Compose has since been removed entirely; see above. Reported by [@walmis](https://github.com/walmis) in [#5](https://github.com/noonghunna/qwen36-27b-single-3090/issues/5).
- **2026-04-27** — `patches/patch_tolist_cudagraph.py` was silently failing on (a) any non-docker setup (hardcoded `dist-packages` path) and (b) any vLLM nightly past the one we initially tested against (multi-line block anchors fragile against upstream rewording). Fixed in [`c34bbf1`](https://github.com/noonghunna/qwen36-27b-single-3090/commit/c34bbf1) — patcher auto-discovers vLLM via `import vllm` and uses single-line regex anchors. Bug reported by [@3dluvr](https://github.com/3dluvr) in [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1).

---

## Requirements

- **GPU:** 1× NVIDIA RTX 3090 (24 GB, Ampere sm_86). Tested; larger cards obviously work too.
- **Driver:** 580.x or newer (for CUDA 13 runtime in the vLLM nightly image).
- **Disk:** ~20 GB free for model weights.
- **Software:**
  - Docker with NVIDIA Container Toolkit
  - `git`, `curl`, `sha256sum` (setup script uses them)
  - `hf` CLI *or* `huggingface-cli` (install: `pip install 'huggingface-hub[hf_transfer]'`)

No system Python required.

---

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/noonghunna/qwen36-27b-single-3090.git
cd qwen36-27b-single-3090

# 2. Fetch Genesis patches + download + SHA-verify the model (~20 GB, 10-30 min)
bash scripts/setup.sh

# 3. Start the server
cd compose && docker compose up -d

# 4. Watch it come up (~2 min for cold compile)
docker logs -f vllm-qwen36-27b
# Wait for "Application startup complete"

# 5. Sanity test
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround",
       "messages":[{"role":"user","content":"Capital of France?"}],
       "max_tokens":30}'

# 6. Run the canonical benchmark
cd .. && bash scripts/bench.sh
```

That's it. The stack serves on `http://localhost:8020/v1/*` as a drop-in OpenAI-compatible endpoint — point any OpenAI SDK, Open WebUI, LM Studio, or Cline at it.

---

## Pick a compose variant

Only one container can bind to port 8020 at a time — `docker compose down` before switching. All variants share the same pinned vLLM image digest. They differ in KV cache dtype, context length, vision, Genesis tree, spec-decode, and (for long-ctx) the `cudagraph_mode=NONE` workaround flag.

```bash
# Default — 20K, vision, tools, fp8 KV, MTP n=3, Genesis  →  55 narr / 70 code TPS
cd compose && docker compose up -d

# Text-only — 75K, no vision, fp8 KV, MTP n=3, Genesis  →  53 narr / 70 code TPS
cd compose && docker compose -f docker-compose.tools-text.yml up -d

# v7.14 — 48K, vision, TQ3 KV, MTP n=3, Genesis P65  →  51 narr / 68 code TPS
#         RECOMMENDED default for tool-using agents. Below GDN forward cliff;
#         all 10 verify-full.sh checks pass. Engine rejects > 48K cleanly.
cd compose && docker compose -f docker-compose.v714.yml up -d

# v7.14 — opt-in 64K-205K (edit v714.yml: change max-model-len + mem-util)
#         See header comment block in v714.yml for the full envelope matrix.
#         All opt-ins at higher ctx have prefill-OOM caveats — read first.

# Long-ctx (DEPRECATED — superseded by v7.14)
# cd compose && docker compose -f docker-compose.longctx-experimental.yml up -d

# No-Genesis MTP — 20K, vision, fp8 KV, MTP n=3, no patches  →  55 narr / 68 code TPS
cd compose && docker compose -f docker-compose.no-genesis-mtp.yml up -d

# Minimal — 32K, vision, fp8 KV, no spec-decode, no patches  →  32 narr / 33 code TPS
cd compose && docker compose -f docker-compose.minimal.yml up -d
```

---

## Production numbers — default config

Measured 2026-04-27 on `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` + Genesis v7.54 (`bf667c7`). 5 measured runs after 3 warmups.

```
  Qwen3.6-27B on 1× RTX 3090 (24 GB, 230W cap, default config)
  ────────────────────────────────────────────────────────────
  wall_TPS         55.0  narrative  (CV 2.9%)   /   70.5  code  (CV 1.2%)
  decode_TPS       55.4  narrative              /   71.8  code
  TTFT             148 ms                       /   147 ms
  Context          20 K tokens
  Vision           Enabled (MoonViT BF16)
  VRAM             22.3 / 24 GB
  Server           vLLM · full OpenAI API
  Tools            ✅ working   Streaming ✅   Thinking ✅
  Spec-decode      MTP n=3
                     narrative — AL 2.62–2.72, accept 78/52/33%
                     code      — AL 3.33–3.45, accept 92/82/68%  (canonical MTP n=3 ceiling)
```

### Cross-rig validation

These numbers are independently reproducible on similar hardware:

| Source | Hardware | vLLM | Genesis | Setup | Narr | Code |
|---|---|---|---|---|---|---|
| **This repo** | 1× 3090 PCIe | dev205 | v7.54 | TP=1, fp8, MTP n=3, 20K | **55.0** | **70.5** |
| **This repo (control)** | 1× 3090 PCIe | dev205 | none | TP=1, fp8, MTP n=3, 20K | 54.7 | 68.2 |
| [u/sudeposutemizligi](https://www.reddit.com/r/LocalLLaMA/) | 2× 3090 PCIe | dev45 | none | TP=2, fp8, MTP n=3, 131K | ~55 (avg of 36/57/54) | ~68 |

Three independent rigs, three different vLLM nightlies, with-and-without Genesis — same converged numbers. **Genesis is performance-neutral on the fp8 + MTP path; the TPS we report is what the hardware delivers.**

---

## Why this works where other recipes don't

Three hurdles had to be cleared for this config to run on a single consumer 24 GB card:

### 1. The published int4-AutoRound quant preserves `mtp.fc` at full precision

A vanilla `auto-round` run on Qwen3.6-27B packs the MTP fusion layer (`mtp.fc`) as INT4. In that form, vLLM's `Qwen3_5MTP` loader silently skips loading it (param name mismatch: expects `fc.weight`, finds `fc.qweight`). Result: MTP "loads" with zero parameters and produces **0% draft acceptance**.

Both [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) and [`Intel/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound) work around this — they ship `mtp.fc.weight` as a plain unquantized BF16 tensor. Lorbus does it implicitly (the `.weight` tensor is in the file with no explicit `extra_config` entry); Intel adds an explicit `mtp.fc: {bits: 16, data_type: fp}` to `extra_config`. Functionally identical: same 18 GB on disk, same 2013 tensors, same architecture, same INT4/group_size=128/auto_round packing. We use Lorbus because it's what we tested end-to-end; Intel's variant should be a drop-in if you prefer that source.

Quick check that whichever quant you use has the fix: look for `mtp.fc.weight` (not `mtp.fc.qweight`) in the safetensors index.

### 2. Genesis patches bypass the TurboQuant hybrid gate

`Qwen3.6-27B` is a Qwen3-Next hybrid model: interleaved DeltaNet (Gated Linear Attention) + standard attention layers. vLLM's TurboQuant KV cache refuses to initialize on hybrid models:

```
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models. Boundary layer protection requires uniform
attention layers.
```

[Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) are a 20-patch runtime monkey-patcher that, among other things, rewrites the hybrid gate to compute boundary protection only over attention layers. Works on Ampere SM 80–86.

### 3. Our `patch_tolist_cudagraph.py` fixes CUDA graph capture

Even with the hybrid gate bypassed, vLLM still crashed during engine warmup:

```
turboquant_attn.py:570  qsl = query_start_loc.tolist()
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

The continuation-prefill branch of `_prefill_attention` forces a GPU→CPU sync via `.tolist()`, which is illegal during CUDA graph capture. This trips when `--speculative-config` + `--enable-chunked-prefill` + `turboquant_*` KV are combined (vLLM PR #40092 — merged 2026-04-23 — fixed the fast path but left this continuation branch untouched).

Our patch (`patches/patch_tolist_cudagraph.py`) is a disk-edit that wraps both `.tolist()` sites with `torch.cuda.is_current_stream_capturing()` guards. During capture, fall back to the graph-safe fast path; at inference, run the original slow path unchanged. Safe because `unified_attention_with_output` is in vLLM V1's `splitting_ops` list — attention outputs during capture are only consulted for memory profiling, not graph content.

Without this patch, the documented workaround is `--compilation-config.cudagraph_mode=none`, which costs **−55% short-prompt TPS** and makes the whole setup net-negative vs plain fp8 KV.

---

## Configuration notes

### Speculative decoding

`num_speculative_tokens=3` (MTP) is the sweet spot on this model. The sweep below was measured on the older substrate (`@sha256:9bba4628a3...`) and is preserved here as a sanity reference for the n-vs-AL pattern; absolute TPS for n=3 on the current pinned image is 55 narrative / 70 code (see "Production numbers — default config" above).

| n | Narr TPS | Code TPS | Mean AL | Position-wise accept |
|---|---|---|---|---|
| 1 | 55 | 59 | 1.9 | 96% |
| 2 | 61 | 70 | 2.4 | 82/56% |
| **3 ⭐** | **55–70** | **70** | **2.6 narr · 3.4 code** | **78/52/33% narr · 92/82/68% code** |
| 4 | 63 | 82 | 3.0 | 83/55/36/**21%** |

n=4 barely beats n=3 on code peak but the position-4 draft accept collapses to 21% — wasted work. Don't go higher.

### KV cache

`turboquant_3bit_nc` is the smallest preset that boots cleanly. `turboquant_4bit_nc` and `turboquant_k8v4` also work with the patches but give less context:

| Preset | Bits | Per-token bytes | Single-card ceiling |
|---|---|---|---|
| default (BF16) | 16 | ~55 KB | ~8K |
| `fp8_e5m2` | 8 | ~28 KB | ~32K |
| `turboquant_k8v4` | 8+4 avg 6 | ~28 KB | ~40K |
| `turboquant_4bit_nc` | 4+4 avg 4 | ~23 KB | ~84K |
| **`turboquant_3bit_nc`** ⭐ | **3+3** | **~17 KB** | **~125K** |

### Context ceiling

`docker-compose.v714.yml` ships with **48K + `gpu-memory-utilization=0.92`** as the production-safe default. Below both prefill cliffs (see "Activation-memory caveat" above), all 10 verify-full.sh checks pass, oversized requests get a clean HTTP 400.

For deployments that need different trade-offs, the full envelope at lower mem-util values (measured 2026-04-28 on dev205 + Genesis v7.54, vision on, TQ3 KV):

| mem-util | Engine ceiling | Activation headroom | Safe single-prompt | Safe tool-prefill |
|---|---|---|---|---|
| **0.92** ⭐ | **~86K** | **~1.9 GB** | **up to 48K (default ships at this max)** | **up to 48K** |
| 0.93 | ~103K | ~1.7 GB | up to ~50K | up to ~50K |
| 0.95 | ~140K | ~1.2 GB | up to ~50K (cliff 2 fires above) | up to ~40K |
| 0.97 | ~175K | ~720 MB | up to ~50K | up to ~16K (ampersandru-class workloads OOM here) |
| 0.98 | ~206K | ~480 MB | up to ~50K | up to ~16K |

The `mem-util` choice **does not** raise Cliff 2 (GDN single-prompt OOM around 50-60K) — that's hardware-bound on a single 24 GB card with this hybrid architecture. Lowering `mem-util` only helps Cliff 1 (TurboQuant tool prefill).

To push to 192K-205K (text only, with both prefill caveats): edit `v714.yml`, set `max-model-len=192000` (or 205000 + uncomment `--language-model-only`) and `gpu-memory-utilization=0.98`. Re-read the activation-memory caveat above before doing so.

Older composes (`longctx-experimental.yml`, deprecated) cap at 125K because `cudagraph_mode=NONE` keeps inductor compiled scratch + a larger eager-decode footprint. v7.14's PIECEWISE downgrade for spec-decode reclaims that headroom.

### Power cap

Production runs at 230W per card (quiet, cool, stable). For ~+10% mean TPS during heavy sessions:

```bash
sudo nvidia-smi -pl 330 -i 0   # replace 0 with your GPU index
```

Past 330W: diminishing returns (SM clocks saturate near 1900 MHz on 3090s).

---

## Benchmarking

```bash
bash scripts/bench.sh
```

Runs both prompts in one invocation: 3 warmup + 5 measured **narrative** (800-word transformer-attention essay, 1000 tokens) + 5 measured **code** (quicksort, 800 tokens). Reports per-prompt wall_TPS / decode_TPS / TTFT mean+std+CV, plus GPU state and the last 3 SpecDecoding metrics lines (mean AL + per-position accept rates).

Use `ONLY=narr` or `ONLY=code` to skip a prompt; override prompts with `PROMPT_NARR` / `PROMPT_CODE`.

Expected numbers on a stock 3090 at 230W (default compose, dev205 + Genesis v7.54):

| Variant | Narr wall_TPS | Code wall_TPS | Code AL |
|---|---|---|---|
| Default | **55.0** ± 1.6 | **70.5** ± 0.8 | 3.4 |
| Tools-text | 53.4 ± 1.4 | 69.6 ± 0.5 | 3.4 |
| v7.14 | 50.5 ± 1.1 | 67.9 ± 1.8 | 3.4 |
| Long-ctx (125K) | 37.9 ± 1.6 | 49.8 ± 0.7 | 3.4 |
| No-Genesis MTP | 54.7 ± 0.6 | 68.2 ± 1.4 | 3.4 |
| Minimal (no spec-dec) | 32.4 ± 0.1 | 32.6 ± 0.2 | n/a |

CVs are 0.4–4.2% across all configs — runs are tight. The 125K variant runs slower than the 20K default because `cudagraph_mode=NONE` keeps inductor compilation on but disables graph capture; the cost is per-forward dispatch overhead.

---

## Evidence matrix — per-test breakdown

Measured on 1× RTX 3090 at 230W cap, vLLM image pinned to tested digest, `scripts/verify-full.sh`:

| Test | Default (MTP + fp8) | `tools-text.yml` (MTP + fp8, no vision) | `longctx-experimental.yml` (MTP + TQ3, cudagraph-off) |
|---|---|---|---|
| Server + Genesis patches | ✅ | ✅ | ✅ |
| Basic completion (Paris) | ✅ | ✅ | ✅ |
| **Tool calling** | **✅** | **✅** | **✅** (verified 2026-04-27 on Genesis v7.54 + dev205 — older "tool-call cascade" warning no longer applies) |
| **Streaming (SSE)** | **✅** clean output | **✅** clean output | **✅** clean output |
| Thinking / reasoning | ✅ | ✅ | ✅ |
| **Long-context recall** (10K) | **✅** | **✅** | **✅** |
| Long-context recall (30K) | n/a (20K cap) | **✅** | **✅** |
| Long-context recall (60K) | n/a | **✅** | **✅** |
| Long-context recall (90K) | n/a | n/a (75K cap) | **✅** |
| Narrative TPS (1000 tok) | **55.0** (CV 2.9%) | 53.4 (CV 2.6%) | 37.9 (CV 4.2%) |
| Code TPS (800 tok quicksort) | **70.5** (CV 1.2%) | 69.6 (CV 0.7%) | 49.8 (CV 1.4%) |
| TTFT (narr / code) | 148 / 147 ms | 148 / 149 ms | 171 / 172 ms |
| Mean AL · accept (code) | 3.33–3.45 · 77–82% | 3.31–3.51 · 77–83% | 3.40–3.49 · 80–83% |
| Max context | 20K | **75K** | **125K** |
| Vision | ✅ | ❌ | ✅ |
| VRAM | 22.3 GB | 22.2 GB | 23.1 GB |

**The original 125K headline (~85–95 TPS) was reproducible only on workloads that don't exercise structured output:** plain narrative or code generation. Tool calls, long-context recall, and streaming all fail catastrophically with cudagraph on under MTP spec-decode. The nine-probe ladder in [Technical background](#technical-background--whats-broken-upstream-and-why-we-work-around-it) below isolates the bug to the CUDA graph capture/replay layer specifically; Triton kernels and torch.compile inductor output are correct when invoked dynamically. The ngram path is now fixed upstream (closed for ngram in #40831 via Sander's v7.13 + #40875); MTP remains tracked at [#40880](https://github.com/vllm-project/vllm/issues/40880).

---

## Technical background — what's broken upstream and why we work around it

**TurboQuant KV is frontier-level.** It landed in vLLM mainline only weeks before this repo was published and is still under active development. The spec-decode × TurboQuant interaction we hit is one of several compatibility edges upstream is still working through (see vLLM's tracking issue [#40069](https://github.com/vllm-project/vllm/issues/40069) — "Speculative decoding / Eagle" and "Hybrid attention models" both unchecked).

Initial symptom: under the originally-shipped 125K config (TurboQuant KV + MTP spec-decode + cudagraph on), the model produces degenerate token loops on tool calls, long-context recall, and occasionally streaming. We isolated the bug through nine probes:

| # | turboquant | spec-dec | cudagraph | torch.compile | result | TPS |
|---|---|---|---|---|---|---|
| 1 | ✅ | off | ✅ | ✅ | ✅ all tests pass | 40 |
| 2 | ✅ | ngram n=3 | ✅ | ✅ | ✗ same loops as MTP | -- |
| 3 (MiMo dense) | ✅ | MTP n=1 | ✅ | ✅ | ✗ first-token collapse | -- |
| 4 | ✅ | MTP n=3 | ✅ | + `_CONTINUATION_DECODE_THRESHOLD=0` | ✗ | -- |
| 5 | ✅ | MTP n=3 | ❌ | ❌ | ✅ all tests pass | 23 |
| **6** | ✅ | MTP n=3 | **❌** | ✅ | **✅ all tests pass** | **33** |
| 7 | ✅ | MTP n=3 (9-prompt structured-output sweep) | ❌ | ✅ | ✅ all 9 prompts pass | 33 |
| 8 | ✅ | MTP n=3 + PR #40798 backport | ✅ | ✅ | ✗ same loops | 96 |
| **9A** | ✅ | MTP n=3 + Genesis v7.13 (#40738 + parser fixes) | ✅ | ✅ | ✗ tool calls fail, recall truncates | -- |
| **9C** | ✅ | ngram n=3 + `prompt_lookup_min=8` + Genesis v7.13 | ✅ | ✅ | ✅ short-ctx clean (filed as cross-confirmation of #40875) | 35 |

**What this isolates:**

- Probe 1 → TurboQuant alone is fine.
- Probes 2-3 → bug isn't MTP-specific; isn't hybrid-attention-specific.
- Probe 4 → bug isn't in the within-batch `_prefill_attention` decode-fast-path routing (paper-backed bias-compounding hypothesis was wrong).
- Probe 5 → disabling **both** torch.compile and cudagraph fixes the bug — compilation machinery is the culprit.
- Probe 6 → disabling **only** cudagraph (keeping torch.compile inductor on) also fixes the bug — **isolating the bug to CUDA graph capture/replay specifically**.
- Probe 7 → confirmed against [Sander's 9-prompt corruption-detection suite](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) (`tool_call_simple`, `code_quicksort`, `structured_xml`, etc.) — all clean.
- Probe 8 → backported PR #40798 (workspace-manager refactor). Bug persists. Buffer-pointer-drift hypothesis was insufficient.
- **Probe 9A → Sander's v7.13 backports (#40738 + parser fixes) do NOT fix MTP × TurboQuant × cudagraph on Qwen3.6-27B.** Filed as [#40880](https://github.com/vllm-project/vllm/issues/40880) — explicitly per Sander's handoff that the v7.13 cycle didn't test MTP at all.
- **Probe 9C → ngram + `prompt_lookup_min=8` + v7.13 backports DO work** at short context (cross-rig + cross-model confirmation of [#40875](https://github.com/vllm-project/vllm/issues/40875)). At ~30K context the stack OOMs due to v7.13's expanded prealloc footprint on a 24 GB card.

**The Triton kernels are correct when invoked dynamically. torch.compile inductor output is correct.** What corrupts the output is how the captured CUDA graph handles spec-decode's runtime shapes vs warmup-shape capture for the TurboQuant attention path. The ngram path is fixed upstream; the MTP path remains open and is what `cudagraph_mode=NONE` works around.

The 125K compose ships `--compilation-config '{"cudagraph_mode":"NONE"}'` as the interim workaround. Cost: ~60% TPS (85 → 33 narrative). Drop the flag once [#40880](https://github.com/vllm-project/vllm/issues/40880) lands.

---

## Troubleshooting

### `Cannot copy between CPU and CUDA tensors during CUDA graph capture`

The `patch_tolist_cudagraph.py` didn't apply. Check the container logs for:

```
[tolist_cudagraph_fix] Patched ... Site A: ok, Site B: ok
```

If not present, the anchor text may have drifted in a newer vLLM nightly. Pin the image in `docker-compose.yml` (change `vllm/vllm-openai:nightly` to a specific tag), or open an issue here.

### `NotImplementedError: TurboQuant KV cache is not supported for hybrid`

Genesis patches didn't apply. Check logs for `INFO:genesis_patch:` lines. Re-run `bash scripts/setup.sh` to ensure `patches/genesis/patch_genesis_unified.py` exists.

### Model load OOMs

- Too little free VRAM at launch. Close other GPU processes.
- If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES=N` in `compose/docker-compose.yml` (uncomment the line).

### `block_size (4128) must be <= max_num_batched_tokens (2048)`

You edited `--max-num-batched-tokens`. Keep it ≥ 4128 for this context length — Qwen3-Next's Mamba block_size scales with `max-model-len`.

### TPS lower than expected

Cross-reference the [Status at a glance](#status-at-a-glance) table for the variant you booted. Common cases:

- **You're on `longctx-experimental.yml` and seeing ~38 narr / 50 code TPS** — expected. `cudagraph_mode=NONE` keeps inductor compilation on but disables graph capture; that's the cost of correctness at 125K. Use Default for max TPS at ≤20K.
- **You're on `minimal.yml` and seeing ~32 TPS** — expected. No spec-decode means no MTP boost.
- **You're on Default and seeing <50 narr or <65 code** — patch likely didn't apply. Check `docker logs vllm-qwen36-27b 2>&1 | grep "Genesis Results"` (should show `26 applied / 37 skipped / 0 failed`) and `grep "tolist_cudagraph"` (should show site A + site B applied).

### Tool calls return `<tool_call>{...}</tool_call>` as plain text (tool extraction doesn't fire)

Two possible causes; the logs distinguish them:

**Cause A — you removed `cudagraph_mode=NONE` from the long-ctx compose.** The original 125K config (cudagraph on) hits [#40831](https://github.com/vllm-project/vllm/issues/40831) and produces `<tool_call>` loops. The shipped `docker-compose.longctx-experimental.yml` already includes the cudagraph-off workaround; if you stripped that flag for performance, restore it.

**Cause B — Genesis patch anchor drift.** Check container logs for:

```
[INFO:genesis.apply_all] Genesis Results: <N> applied, <M> skipped, <K> failed
```

If `failed > 0`, your vLLM image drifted past anchors Genesis expects. All composes pin to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (= vLLM `0.19.2rc1.dev205+g07351e088`, Sandermage's documented Genesis test target). Genesis v7.54 lands clean on this pin (`Genesis Results: 27 applied / 36 skipped / 0 failed` for TQ paths, `26/37/0` for fp8 paths) — verified 2026-04-27. To verify patch applicability against any image:

```bash
docker run --rm --entrypoint python3 \
  -v $(pwd)/patches/genesis/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08 \
  -m vllm._genesis.patches.apply_all 2>&1 | grep -E "applied|FAILED|skipped" | tail -10
```

---

## Repo layout

```
qwen36-27b-single-3090/
├── README.md                                   (this file)
├── LICENSE                                     Apache-2.0
├── .gitignore
├── patches/
│   ├── patch_tolist_cudagraph.py               our CUDA graph capture crash fix (#40807)
│   ├── patch_pr40798_workspace.py              research artifact — backports vllm#40798;
│   │                                            does NOT fix #40831 (probe 8); kept for
│   │                                            reproducibility of the negative result
│   └── genesis/                                (gitignored; fetched by setup.sh)
├── compose/
│   ├── docker-compose.yml                      DEFAULT — fp8 + MTP + vision + Genesis, 20K, 55 narr / 70 code TPS
│   ├── docker-compose.tools-text.yml           fp8 + MTP + Genesis, no vision, 75K, 53 narr / 70 code TPS
│   ├── docker-compose.v714.yml                 ⭐ TQ3 + MTP + Genesis P65, 48K default vision (opt-in 64K-205K), 51 narr / 68 code TPS
│   ├── docker-compose.no-genesis-mtp.yml       fp8 + MTP, no patches, 20K, 55 narr / 68 code TPS
│   ├── docker-compose.minimal.yml              fp8, no spec-decode, 32K, 32 narr / 33 code TPS
│   └── docker-compose.longctx-experimental.yml DEPRECATED — superseded by v714 (default 48K, opt-in to higher)
└── scripts/
    ├── setup.sh                                clone Genesis + download model + SHA verify
    ├── verify.sh                               quick smoke test (~10 sec)
    ├── verify-full.sh                          full functional test — streaming, thinking, needle (~3 min)
    └── bench.sh                                canonical TPS bench
```

---

## What this is NOT

- A vLLM fork — `patch_tolist_cudagraph.py` is a disk-edit applied at container startup, not a fork. When upstream merges the fix, this patch becomes a no-op (anchor won't match, script prints a warning and exits cleanly).
- A quantization recipe — we use Lorbus's INT4 quant as-is. The recipe for producing future `mtp.fc`-preserved quants is in [Lorbus's model card](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound#reproduction).
- A benchmark rig — included `bench.sh` is the minimum needed to verify your setup matches ours. For rigorous A/B comparisons use something like [`vllm-project/bench`](https://github.com/vllm-project/bench).

---

## Upstream status

- **[#40069](https://github.com/vllm-project/vllm/issues/40069)** — TurboQuant/HIGGS follow-ups tracker (upstream). Lists "Speculative decoding / Eagle" and "Hybrid attention models" as unchecked.
- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — our CUDA graph `.tolist()` crash; worked around locally via `patch_tolist_cudagraph.py`. Sandermage's [v7.10 Genesis tree](https://github.com/Sandermage/genesis-vllm-patches) reaches the same end state via pre-allocation (Patches 23 + 44).
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — our TurboQuant × spec-decode output-quality bug. **Closed for the ngram path** via Sander's v7.13 backports of upstream PRs (#40738 GDN state recovery + #36138 + #40783 + #39055) plus the [#40875](https://github.com/vllm-project/vllm/issues/40875) `prompt_lookup_min=8` config trick. **MTP path remains broken** — tracked at [#40880](https://github.com/vllm-project/vllm/issues/40880) (see below).
- **[#40875](https://github.com/vllm-project/vllm/issues/40875)** — Sander's follow-up identifying that `prompt_lookup_min=2` (default) causes ngram to find spurious matches in chat-template tool definitions. Setting `prompt_lookup_min=8` is a config-only fix, validated on Sander's 35B-A3B and confirmed on our 27B (probe 9 Test C). For ngram users, this + v7.13 backports = working stack with cudagraph ON.
- **[#40880](https://github.com/vllm-project/vllm/issues/40880)** — our MTP-specific follow-up filed at Sander's [explicit handoff](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4319965017): *"we did not test MTP at all in the v7.13 cycle... your data shows that assumption is wrong."* MTP × TurboQuant × cudagraph remains broken even with all v7.13 backports applied (probe 9 Test A); the four upstream PRs scope to ngram's GDN state recovery and don't cover the Eagle/MTP forward path. **Our `cudagraph_mode=NONE` workaround in `docker-compose.longctx-experimental.yml` stays in place until this lands.**
- **[PR #40798](https://github.com/vllm-project/vllm/pull/40798)** — *hypothesized fix that didn't pan out.* Moves `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf` from per-layer `register_buffer(B=max_num_seqs)` to `WorkspaceManager.get_simultaneous()`. Sander and I both expected this would close the pointer-drift between warmup-shape capture and runtime-shape replay. Probe 8 backported the full PR diff via [`patches/patch_pr40798_workspace.py`](./patches/patch_pr40798_workspace.py) (research artifact, not shipped) and the bug persisted. Useful negative result documented on the PR thread.
- **Sandermage's [P56](https://github.com/Sandermage/genesis-vllm-patches/blob/main/vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py)** — routing-layer workaround (architecturally equivalent to our Probe 4 patch). Marked superseded by our `cudagraph_mode=NONE` workaround since it only addresses the catastrophic surface.
- Sandermage Genesis: we may contribute `patch_tolist_cudagraph.py` to their unified script. They have offered to extract Patches 23 + 44 to upstream.

Until upstream lands a fix: fp8_e5m2 + MTP at 20K (default) is the fast option, cudagraph-off + turboquant + MTP at 125K (long-ctx variant) is the long-context option. Both fully functional. The cudagraph-off variant pays a ~60% TPS cost; that recovers when the underlying bug is fixed.

---

## Credits

- **Qwen team** (@Alibaba_Qwen) — for the base model and a usable MTP head architecture
- **Lorbus** — for the AutoRound INT4 quant with preserved BF16 `mtp.fc`
- **[Sandermage](https://github.com/Sandermage/genesis-vllm-patches)** — for the Genesis patch set that made TurboQuant work on hybrid models on consumer Ampere; for independently reproducing #40831 on a different rig (2× A5000 + Qwen3-Next-35B-A3B-FP8 + ngram), confirming the cudagraph-off workaround, and engaging honestly with each negative result during the probe ladder
- **[vibhavagarwal5](https://github.com/vllm-project/vllm/pull/38479)** — for the original TurboQuant landing PR and the [tracking issue #40069](https://github.com/vllm-project/vllm/issues/40069) that made the spec-decode-unverified status visible upfront
- **vLLM project** — for shipping TurboQuant and actively maintaining the backend
- **Intel AutoRound** — for the quantization framework

Our contribution here is `patch_tolist_cudagraph.py`, the original write-up linking it all together, and this reproducible recipe. Everything else is brilliant work by people we stand on the shoulders of.

---

## License

Apache 2.0. Do what you want with it. If you get better numbers, please open an issue — we'd love to see it.
