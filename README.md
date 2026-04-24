# Qwen3.6-27B on a single RTX 3090

**92 TPS narrative · 95 TPS code · 125K context · vision enabled · 230W cap**

A reproducible recipe for serving [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) on a single consumer 24 GB RTX 3090, via vLLM with MTP speculative decoding + TurboQuant 3-bit KV cache. Built on top of [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) plus a CUDA graph capture fix for vLLM's TurboQuant backend that ships in this repo.

> 📖 **Full write-up:** *[Qwen3.6-27B on a single RTX 3090 — the recipe](https://medium.com/)*  *(replace with live URL after publishing)*
> 🐛 **Upstream bug report:** [vllm-project/vllm#40807](https://github.com/vllm-project/vllm/issues/40807)

---

## The numbers

```
  Qwen3.6-27B on 1× RTX 3090 (24 GB, 230W cap)
  ─────────────────────────────────────────────
  Throughput      92 TPS (narrative)  /  95 TPS (code, peak 96)
  Context          125 K tokens (KV pool 198K)
  Vision           Enabled (MoonViT BF16)
  VRAM            21.3 / 24 GB
  Server          vLLM · full OpenAI API
  Tools           ✅    Prefix cache  ✅
  Spec-decode    MTP n=3 · AL 3.4–3.8 · accept 97/95/91%
```

For comparison:

| Config | Narrative TPS | Code TPS | Context |
|---|---|---|---|
| vLLM + fp8 KV + MTP n=3 (earlier 3090 config) | 63.8 | 79.7 | 20 K |
| [Lorbus card reference](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (RTX 5090) | ~60 | ~60 | 262 K |
| **This repo (1× RTX 3090)** | **91.9 mean / 95.3 peak** | **94.6 mean / 95.9 peak** | **125 K** |
| llama.cpp mainline + q4_0 KV (single-card long-ctx fallback) | 28.5 | 28.4 | 262 K |

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

**High variance is normal** — MTP acceptance dips cause individual requests to drop into the 60–70 TPS range. Mean over 3+ runs is the honest number (~85 TPS).

---

## Known issue: tool calling × MTP × TurboQuant KV

**Status:** diagnosed 2026-04-24 via a compose sweep, fix available via alternate compose file.

### Symptom

Requests with a `tools: [...]` array produce degenerate output — `<tool_call>` repeated hundreds of times with no JSON body, sometimes `I I I I` token loops. The `tool_calls[]` field in the response stays empty. Run `scripts/verify.sh` to detect.

### Root cause

Not the Genesis tool_call fix (that applies correctly), not a model weight issue. **MTP speculative decoding × any TurboQuant KV preset** is incompatible on tool-schema prompts. Other prompt classes (narrative, code, vision, long context) are fine; only structured tool-call prompts trigger the collapse.

Sweep results (MTP n=3 in all tests where MTP is on):

| KV preset | Tools work? |
|---|---|
| `fp8_e5m2` | ✅ |
| `turboquant_3bit_nc` | ❌ |
| `turboquant_4bit_nc` | ❌ |
| `turboquant_k8v4` (no norm correction) | ❌ |
| Any TurboQuant, MTP **disabled** | ✅ |

The MTP draft head's attention computation interacts badly with TurboQuant's KV storage format on out-of-distribution token sequences. Draft acceptance collapses to 0%, and the main model's own samples also degenerate (probably because earlier garbage tokens contaminate the context).

### Pick a variant based on what you need

All three benched on a single RTX 3090 at 230W cap, vLLM image pinned to the tested digest, 3× warmup + 3× narrative (1000 tok) + 2× code (800 tok) runs.

| If you need... | Use | Ctx | Narr TPS | Code TPS | Tools | VRAM |
|---|---|---|---|---|---|---|
| **Max context + vision, no tools** (the article's headline) | `docker compose up -d` (default `docker-compose.yml`) | **125K** | **91.9** | **94.6** (peak 95.9) | ❌ | 22.0 GB |
| **Tool calling + reasonable speed** | `docker compose -f docker-compose.tools.yml up -d` | 20K | 65.9 | 84.4 (peak 85.2) | ✅ | 22.8 GB |
| **Tool calling + max context (slower)** | Copy `docker-compose.tools.yml`, set `--kv-cache-dtype turboquant_3bit_nc`, set `--max-model-len 125000`, remove the `--speculative-config` line | 125K | 39.8 | 39.7 | ✅ | 22.0 GB |

**Interpreting the code/narr split:** configs with MTP spec-decode produce faster code than narrative (code is more predictable → higher MTP accept → more tokens per verify step). Without MTP (config 3), the two converge around 40 TPS because decode is no longer draft-accelerated.

Only one container can bind to port 8020 — `docker compose down` before switching.

Measured 2026-04-24 on `vllm/vllm-openai@sha256:9bba4628a3b9...` (digest pinned in both compose files).

### Upstream-bug potential

This looks like a real vLLM/TurboQuant bug — MTP and TurboQuant should compose correctly regardless of prompt class. Likely to be filed upstream after a second round of validation (reproduces independently of our patch, on plain `vllm/vllm-openai:nightly` with Genesis). Watch the repo for updates.

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

Check container logs for this line from the Genesis patcher:

```
[11/17] Qwen3 <tool_call> implicit reasoning end (PR #35687)...
  [FAILED] Qwen3 tool_call fix
```

If you see `[FAILED]`, you're running a vLLM nightly that drifted past the anchor Genesis Patch 12 expects. Qwen3 emits `<think>...<tool_call>{...}</tool_call>` often **without** closing `</think>` first; without the patch, the reasoning parser eats the whole output as reasoning and never extracts the tool call → client sees `<tool_call>` as literal text.

**Fix:** pin the image to the exact nightly we tested (already pinned by default in `compose/docker-compose.yml`):

```yaml
image: vllm/vllm-openai@sha256:9bba4628a3b943e0dd33caefb94b811569ba1e97bdf23bee19a265c31b947ccb
```

On that digest (vLLM `0.19.2rc1.dev21+g893611813`, built 2026-04-20), all four Qwen3 tool-call sub-patches apply cleanly — look for `[OK] Qwen3 tool_call fix` in the logs to confirm.

If you need the floating `:nightly` tag for other reasons, verify the patch applied before trusting tool calls. You can force-replay the patch against a fresh container to see which anchor drifted:

```bash
docker run --rm --entrypoint python3 vllm/vllm-openai:nightly \
  /patches/patch_genesis_unified.py 2>&1 | grep -E "Patch|FAILED|OK"
```

---

## Repo layout

```
qwen36-27b-single-3090/
├── README.md                            (this file)
├── LICENSE                              Apache-2.0
├── .gitignore
├── patches/
│   ├── patch_tolist_cudagraph.py        our CUDA graph capture fix
│   └── genesis/                         (gitignored; fetched by setup.sh)
├── compose/
│   └── docker-compose.yml               T5 production config
└── scripts/
    ├── setup.sh                         clone Genesis + download model + SHA verify
    └── bench.sh                         canonical TPS bench
```

---

## What this is NOT

- A vLLM fork — `patch_tolist_cudagraph.py` is a disk-edit applied at container startup, not a fork. When upstream merges the fix, this patch becomes a no-op (anchor won't match, script prints a warning and exits cleanly).
- A quantization recipe — we use Lorbus's INT4 quant as-is. The recipe for producing future `mtp.fc`-preserved quants is in [Lorbus's model card](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound#reproduction).
- A benchmark rig — included `bench.sh` is the minimum needed to verify your setup matches ours. For rigorous A/B comparisons use something like [`vllm-project/bench`](https://github.com/vllm-project/bench).

---

## Upstream status

- vLLM issue: tracked under [#40069](https://github.com/vllm-project/vllm/issues/40069) (TurboQuant/HIGGS follow-ups). Our specific `.tolist()` bug is filed as [#40807](https://github.com/vllm-project/vllm/issues/40807).
- vLLM PR: not yet submitted — our disk-edit is the short-term workaround.
- Sandermage Genesis: we may contribute `patch_tolist_cudagraph.py` upstream as a new patch in their unified script.
- Lorbus HF discussion: caveat about the Ampere cudagraph requirement posted at TBD.

Once upstream vLLM handles the sync properly (pin-memory or precompute), this repo becomes a historical curiosity. Until then: copy-paste and go.

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
