# Qwen3.6-27B on a single RTX 3090

**A validated recipe for serving Qwen3.6-27B on a single consumer 24 GB RTX 3090** — full OpenAI API, vision, tool calling, streaming, speculative decoding, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via vLLM with MTP speculative decoding + fp8_e5m2 KV cache. Built on [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) + a CUDA graph capture fix that ships in this repo.

> 📖 **Write-up:** *[Qwen3.6-27B on a single RTX 3090 — the recipe](https://medium.com/)*
> 🐛 **Upstream bug reports:** [vllm-project/vllm#40807](https://github.com/vllm-project/vllm/issues/40807) (CUDA graph crash — worked around locally) · [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831) (TurboQuant × any spec-decode output-quality bug — confirmed universal)

---

## ⚠ About the 125K context headline

The [original write-up](https://medium.com/) reported 85–106 TPS at **125K context** using TurboQuant KV cache. Under broader functional testing since publication, we found that **TurboQuant KV × any speculative decoding method (MTP *or* ngram) has output-quality issues** on prompts that require structured or exact output:

- Tool calls: `<tool_call>` token loops, empty `tool_calls[]`
- Long-context recall: model outputs first token then loops (`amber amber amber...`)
- Streaming: occasional degenerate loops on some free-form completions

The 125K **KV pool** fits; the **attention output quality** is unreliable under that config. Short narrative/code completions (what the benchmark measured) work fine; anything requiring recall or structured output doesn't.

**We traced the failure through three probes:**

1. TurboQuant with **MTP disabled** on the same model → every test passes cleanly. TurboQuant alone is not the bug.
2. TurboQuant with **ngram speculative decoding** (no neural draft head) on the same model → same degenerate-loop failure shape. Rules out "the MTP draft head reading quantized KV" as the hypothesis.
3. TurboQuant + MTP on **MiMo-7B-RL** (pure dense attention, no DeltaNet) → *catastrophic* first-token collapse. Rules out "hybrid-attention-specific bug."

The bug is in the **TurboQuant attention backend under any spec-decode's multi-token verify/rollback pattern**, independent of draft method and independent of attention architecture. Severity scales with the fraction of full-attention layers: Qwen3.6 hybrid (25% full-attn) limps on structured outputs; MiMo dense (100% full-attn) collapses from token 1.

**TurboQuant KV is a genuinely frontier-level KV compression.** It's been in vLLM mainline for only weeks and several open bugs show the backend is still being hardened around non-trivial write patterns (long prefill, high concurrency, hybrid layer geometry). The spec-decode interaction is another one of those edges. We've filed upstream as [#40831](https://github.com/vllm-project/vllm/issues/40831) with the full isolation matrix.

**This repo's *default* config is now what we validated end-to-end:** MTP n=3 + **fp8_e5m2 KV** (not TurboQuant) + vision — 20K ctx, tools + streaming + recall all working. The 125K TurboQuant config is still available as `docker-compose.longctx-experimental.yml` for users who want the original headline behavior and understand the caveats.

---

## Evidence matrix — what works on each config

Measured on 1× RTX 3090 at 230W cap, vLLM image pinned to tested digest, `scripts/verify-full.sh`:

| Test | Default (MTP + fp8) | `tools-text.yml` (MTP + fp8 + no vision) | `longctx-experimental.yml` (MTP + turboquant) |
|---|---|---|---|
| Server + Genesis patches | ✅ | ✅ | ✅ |
| Basic completion (Paris) | ✅ | ✅ | ✅ |
| **Tool calling** | **✅** | **✅** | ❌ `<tool_call>` loops |
| **Streaming (SSE)** | **✅** clean output | **✅** clean output | ❌ degenerate loops on some prompts |
| Thinking / reasoning | ✅ | ✅ | ✅ (slow to finish — Qwen3 is verbose) |
| **Long-context recall** (10K) | **✅** | **✅** | ❌ first token + loop |
| Long-context recall (30K) | n/a (20K cap) | **✅** | ❌ |
| Long-context recall (60K) | n/a | **✅** | ❌ |
| Short-prompt TPS (narr / code) | 65.9 / 84.4 | 65.2 / 83.8 | 91.9 / 94.6 |
| Peak TPS | 85 | 85 | 96 |
| Max context | 20K | **75K** | 125K (KV pool, attention unreliable) |
| Vision | ✅ | ❌ | ✅ |
| VRAM | 22.8 GB | 22.2 GB | 22.0 GB |

**Root cause of the ❌ cells:** Any speculative decoding method combined with any TurboQuant KV preset produces degenerate token loops. Tested combinations that all fail: MTP (n=1, 2, 3, 4) × `turboquant_3bit_nc` / `turboquant_4bit_nc` / `turboquant_k8v4`, and ngram spec-decode × `turboquant_3bit_nc`. Also independently verified on **MiMo-7B-RL** (dense-attention non-hybrid model — collapses from the first generated token). Either spec-decode alone (on fp8 KV) or TurboQuant alone (spec-decode off) works fine — only the combination fails. Filed upstream as [#40831](https://github.com/vllm-project/vllm/issues/40831) with full isolation matrix (separate from #40807, which is the CUDA graph crash we work around locally).

---

## Production numbers — default config

```
  Qwen3.6-27B on 1× RTX 3090 (24 GB, 230W cap, default config)
  ────────────────────────────────────────────────────────────
  Throughput      66 TPS (narrative)  /  84 TPS (code, peak 85)
  Context          20 K tokens
  Vision           Enabled (MoonViT BF16)
  VRAM            22.8 / 24 GB
  Server          vLLM · full OpenAI API
  Tools           ✅ working   Streaming ✅   Thinking ✅
  Spec-decode    MTP n=3 · AL 2.87–3.39 · accept 94/81/64%
```

Beats [Lorbus card's RTX 5090 baseline](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (~60 TPS) on consumer Ampere hardware. All functionality verified.

For higher TPS at the cost of broken tools and unreliable recall, see `docker-compose.longctx-experimental.yml` (92/95 TPS, 125K ctx pool, quality caveats documented in file header).

For more context without vision (text-only agents), see `docker-compose.tools-text.yml` (75K ctx, same TPS).

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

## Why this works where other recipes don't

Three hurdles had to be cleared for this config to run on a single consumer 24 GB card:

### 1. Lorbus's quant preserves `mtp.fc` as BF16

A vanilla `auto-round` run on Qwen3.6-27B packs the MTP fusion layer (`mtp.fc`) as INT4. In that form, vLLM's `Qwen3_5MTP` loader silently skips loading it (param name mismatch: expects `fc.weight`, finds `fc.qweight`). Result: MTP "loads" with zero parameters and produces **0% draft acceptance**.

Lorbus's AutoRound release [dequantizes `mtp.fc` back to BF16 after quantization finishes](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound#mtp-fix--whats-different-from-a-vanilla-autoround-run). Adds ~100 MB, unlocks the full MTP speedup.

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

`num_speculative_tokens=3` (MTP) is the sweet spot on this model:

| n | Narr TPS | Code TPS | Mean AL | Position-wise accept |
|---|---|---|---|---|
| 1 | 55 | 59 | 1.9 | 96% |
| 2 | 61 | 70 | 2.4 | 82/56% |
| **3 ⭐** | **64** | **80** | **3.4** | **92/81/67%** |
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

`--max-model-len 125000` is the largest value vLLM's `_check_enough_kv_cache_memory` pre-check accepts with our config. The KV pool itself (198K tokens) is larger, but the extra capacity is used for Mamba/DeltaNet recurrent state + prefix cache + spec-decode scratch blocks — **don't bypass the pre-check**, those tokens are load-bearing.

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

Runs 3 warmup + 3 narrative (800-word essay, 1000 tokens) + 2 code (quicksort, 800 tokens) against the canonical prompts used throughout this repo. Reports wall time, completion tokens, TPS per request, plus GPU state and the last 3 SpecDecoding metrics lines (mean AL + per-position accept rates).

Expected numbers on a stock 3090 at 230W:

| Run | Wall | TPS |
|---|---|---|
| warmup 1 (cold) | 12–15 s | 70–80 |
| warmup 2+3 (warm) | 10–11 s | 90–100 |
| narrative (warmed) | 10–16 s | 60–105 |
| code (warmed) | 8–12 s | 60–100 |

**High variance on the experimental 125K config is normal** — MTP acceptance dips cause individual requests to drop into the 60–70 TPS range on the TurboQuant variant. Default + tools-text are much more consistent (fp8 KV doesn't have the same variance profile).

---

## Pick a compose variant

| Workload | Compose file | Context |
|---|---|---|
| **Default — vision + tools + 20K** (validated end-to-end) | `docker-compose.yml` | 20K |
| **Text-only agents + 75K ctx** (drops vision) | `docker-compose.tools-text.yml` | 75K |
| **⚠ Experimental — 125K pool, TurboQuant-only** (tools broken, recall loops) | `docker-compose.longctx-experimental.yml` | 125K KV pool |

Only one container can bind to port 8020 at a time — `docker compose down` before switching.

All three compose files use the same pinned vLLM image digest, the same Genesis patches, and the same MTP n=3 spec-decode. The only differences are `--kv-cache-dtype`, `--max-model-len`, and whether `--language-model-only` is set.

---

## Technical background — why turboquant is on the experimental shelf

**TurboQuant KV is frontier-level.** It landed in vLLM mainline only weeks before this repo was published and is still under active development. The spec-decode × TurboQuant interaction we hit is one of several compatibility edges that upstream is still working through (see vLLM's tracking issue [#40069](https://github.com/vllm-project/vllm/issues/40069) — "Speculative decoding / Eagle" and "Hybrid attention models" are both still unchecked boxes, and our [#40831](https://github.com/vllm-project/vllm/issues/40831) is a concrete repro of both).

What we observed across three probes on two different models:

| Model | Attention | KV | Spec-decode | Structured outputs |
|---|---|---|---|---|
| Qwen3.6-27B | hybrid (16/64 full-attn) | `turboquant_3bit_nc` | MTP n=3 | ❌ degenerate token loops |
| Qwen3.6-27B | hybrid | `turboquant_4bit_nc` | MTP n=3 | ❌ same |
| Qwen3.6-27B | hybrid | `turboquant_k8v4` | MTP n=3 | ❌ same |
| Qwen3.6-27B | hybrid | `turboquant_3bit_nc` | MTP n=1 | ❌ same (not N-dependent) |
| Qwen3.6-27B | hybrid | `turboquant_3bit_nc` | **ngram n=3** | ❌ **same shape — not MTP-specific** |
| **MiMo-7B-RL** | **dense (36/36)** | `turboquant_3bit_nc` | MTP n=1 | ❌ **total collapse from token 1** |
| MiMo-7B-RL | dense | `turboquant_3bit_nc` | **off** | ✅ fully coherent |
| Qwen3.6-27B | hybrid | `turboquant_3bit_nc` | **off** | ✅ all tests pass |
| Qwen3.6-27B | hybrid | **`fp8_e5m2`** | MTP n=3 | ✅ all tests pass |

Earlier write-ups speculated this was an "MTP draft head reading quantized KV" issue. The ngram result (no neural draft) and the MiMo dense-attention result disprove that hypothesis. The bug is in the **TurboQuant attention backend under any spec-decode's multi-token verify/rollback/re-read pattern** — independent of draft method and independent of attention architecture. Probable culprit is one of: speculative writes not tagged as tentative in the store path; decode kernel returning non-deterministic dequant across redundant reads; or rollback state not propagating when a draft is rejected.

Severity scales with the fraction of full-attention layers TurboQuant quantizes: Qwen3.6 (25% full-attn) limps on structured outputs but plain narrative sometimes limps through; MiMo (100% full-attn) collapses on the first generated token regardless of prompt.

**What we're doing about it:**

- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — CUDA graph crash we worked around with `patches/patch_tolist_cudagraph.py`
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — output-quality bug filed with full isolation matrix; comments cross-reference adjacent PRs ([#40074](https://github.com/vllm-project/vllm/pull/40074), [#40122](https://github.com/vllm-project/vllm/pull/40122), [#40706](https://github.com/vllm-project/vllm/pull/40706), [#40798](https://github.com/vllm-project/vllm/pull/40798)) that are touching the same backend paths
- The repo's default will stay on fp8_e5m2 until upstream has a tested fix

When TurboQuant gets proper spec-decode compatibility in mainline, the default will swap back to a long-ctx config and we'll update this README + article accordingly. Until then, treat `turboquant_*` KV presets combined with any speculative decoding as cutting-edge with known rough edges.

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

### Short-prompt TPS stuck at ~30

You probably have `--compilation-config.cudagraph_mode=none` somewhere. Remove it — our patch fixes the underlying bug, cudagraphs should stay on.

### Tool calls return `<tool_call>{...}</tool_call>` as plain text (tool extraction doesn't fire)

Two possible causes; the logs distinguish them:

**Cause A — you're running the experimental compose** (`docker-compose.longctx-experimental.yml`). Tool calls are **known broken** on that config due to TurboQuant KV × spec-decode incompatibility (confirmed on both MTP and ngram draft methods, independent of model architecture — see "Technical background" section above and the file header of the experimental compose). The fix is to use the default `docker-compose.yml` (or `docker-compose.tools-text.yml` if you need more context).

**Cause B — Genesis patch anchor drift.** Check container logs for:

```
[11/17] Qwen3 <tool_call> implicit reasoning end (PR #35687)...
  [FAILED] Qwen3 tool_call fix
```

If you see `[FAILED]`, your vLLM image drifted past the anchor Genesis Patch 12 expects. Pin to our tested digest (already pinned by default in all compose files):

```yaml
image: vllm/vllm-openai@sha256:9bba4628a3b943e0dd33caefb94b811569ba1e97bdf23bee19a265c31b947ccb
```

On that digest (vLLM `0.19.2rc1.dev21+g893611813`, built 2026-04-20), all four Qwen3 tool-call sub-patches apply cleanly — look for `[OK] Qwen3 tool_call fix`. To verify patch applicability against any image:

```bash
docker run --rm --entrypoint python3 vllm/vllm-openai:nightly \
  /patches/patch_genesis_unified.py 2>&1 | grep -E "Patch|FAILED|OK"
```

---

## Repo layout

```
qwen36-27b-single-3090/
├── README.md                                   (this file)
├── LICENSE                                     Apache-2.0
├── .gitignore
├── patches/
│   ├── patch_tolist_cudagraph.py               our CUDA graph capture fix
│   └── genesis/                                (gitignored; fetched by setup.sh)
├── compose/
│   ├── docker-compose.yml                      DEFAULT — MTP + fp8 + vision, 20K
│   ├── docker-compose.tools-text.yml           text-only, 75K ctx
│   └── docker-compose.longctx-experimental.yml 125K pool — ⚠ quality caveats in file
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
- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — our CUDA graph `.tolist()` bug; worked around locally via `patch_tolist_cudagraph.py`.
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — our TurboQuant × any-spec-decode output-quality bug; confirmed across both MTP and ngram, and across hybrid (Qwen3.6) and dense (MiMo-7B) attention. No local workaround; use `fp8_e5m2` KV instead of `turboquant_*` until fixed.
- vLLM PR: not yet submitted — `patch_tolist_cudagraph.py` is the short-term workaround for #40807; #40831 needs an upstream fix in the TurboQuant backend.
- Sandermage Genesis: we may contribute `patch_tolist_cudagraph.py` upstream as a new patch in their unified script.
- Lorbus HF discussion: caveat about the Ampere cudagraph requirement posted at TBD.

Once upstream vLLM handles both issues, this repo becomes a historical curiosity and the default compose can move to TurboQuant KV for the full 125K pool. Until then: fp8_e5m2 + MTP is the validated sweet spot.

---

## Credits

- **Qwen team** (@Alibaba_Qwen) — for the base model and a usable MTP head architecture
- **Lorbus** — for the AutoRound INT4 quant with preserved BF16 `mtp.fc`
- **Sandermage** — for the Genesis patch set that made TurboQuant work on hybrid models
- **vLLM project** — for shipping TurboQuant and actively maintaining the backend
- **Intel AutoRound** — for the quantization framework

Our contribution here is `patch_tolist_cudagraph.py`, the article linking it all together, and this reproducible recipe. Everything else is brilliant work by people we stand on the shoulders of.

---

## License

Apache 2.0. Do what you want with it. If you get better numbers, please open an issue — we'd love to see it.
