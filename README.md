# Qwen3.6-27B on a single RTX 3090

**A validated recipe for serving Qwen3.6-27B on a single consumer 24 GB RTX 3090** — full OpenAI API, vision, tool calling, streaming, speculative decoding, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via vLLM with MTP speculative decoding + fp8_e5m2 KV cache. Built on [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) + a CUDA graph capture fix that ships in this repo.

> 📖 **Write-up:** *[Qwen3.6-27B on a single RTX 3090 — the recipe](https://medium.com/)*
> 🐛 **Upstream bug reports:** [vllm-project/vllm#40807](https://github.com/vllm-project/vllm/issues/40807) (CUDA graph crash — worked around locally) · [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831) (TurboQuant × spec-decode output corruption — closed for ngram path via @Sandermage's v7.13 + [#40875](https://github.com/vllm-project/vllm/issues/40875) `prompt_lookup_min=8` config) · [vllm-project/vllm#40880](https://github.com/vllm-project/vllm/issues/40880) (MTP × TurboQuant × cudagraph — open, our `cudagraph_mode=NONE` workaround stands)

---

## Status at a glance

Three configurations, **all functional today**. Pick the one that matches your workload.

| Variant | Context | TPS (narrative / peak) | Vision | Caveats |
|---|---|---|---|---|
| **Default** (`docker-compose.yml`) | 20K | 66 / 85 | ✅ | None — recommended for most workloads |
| **Tools-text** (`docker-compose.tools-text.yml`) | 75K | 65 / 85 | ❌ | Drops vision to free KV pool |
| **Long-context** (`docker-compose.longctx-experimental.yml`) | 125K | 33 sustained | ✅ | ~60% TPS cost — ships `cudagraph_mode=NONE` as a workaround for [vllm#40831](https://github.com/vllm-project/vllm/issues/40831); flag drops out + TPS recovers when upstream lands a fix |

### What's NOT working today

- **125K context at full TPS (~85+)** — blocked on upstream cudagraph-capture bug [vllm#40831](https://github.com/vllm-project/vllm/issues/40831). Root cause within the cudagraph layer is still TBD upstream. Our long-context variant ships a working workaround at the cost of ~60% TPS; details in [Technical background](#technical-background--whats-broken-upstream-and-why-we-work-around-it) below.
- **GGUF on vLLM for Qwen3-Next family** — not supported upstream yet (4 PRs open + a missing port of `Qwen35TensorProcessor` value transforms). Use llama.cpp / Ollama if you specifically need GGUF.

**TL;DR:** if you only need ≤20K context, run the default and ignore the rest of this README — it works.

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

Only one container can bind to port 8020 at a time — `docker compose down` before switching. All three variants share the same pinned vLLM image digest, the same Genesis patches, the same MTP n=3 spec-decode, and the same `patch_tolist_cudagraph.py`. They differ only in `--kv-cache-dtype`, `--max-model-len`, whether `--language-model-only` is set, and (for the long-ctx variant) the `cudagraph_mode=NONE` workaround flag.

```bash
# Default — 20K, vision, tools, ~85 TPS
cd compose && docker compose up -d

# Text-only at 75K
cd compose && docker compose -f docker-compose.tools-text.yml up -d

# Long-context at 125K (cudagraph-off workaround, ~33 TPS)
cd compose && docker compose -f docker-compose.longctx-experimental.yml up -d
```

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

Beats [Lorbus card's RTX 5090 baseline](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (~60 TPS) on consumer Ampere hardware.

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

The 125K variant runs at a more uniform ~33 TPS because it disables CUDA graph capture (`cudagraph_mode=NONE`) while keeping torch.compile inductor on; spec-decode acceptance dips don't compound with cudagraph variance.

---

## Evidence matrix — per-test breakdown

Measured on 1× RTX 3090 at 230W cap, vLLM image pinned to tested digest, `scripts/verify-full.sh`:

| Test | Default (MTP + fp8) | `tools-text.yml` (MTP + fp8 + no vision) | `longctx-experimental.yml` (MTP + turboquant + cudagraph-off) |
|---|---|---|---|
| Server + Genesis patches | ✅ | ✅ | ✅ |
| Basic completion (Paris) | ✅ | ✅ | ✅ |
| **Tool calling** | **✅** | **✅** | **✅** |
| **Streaming (SSE)** | **✅** clean output | **✅** clean output | **✅** clean output |
| Thinking / reasoning | ✅ | ✅ | ✅ |
| **Long-context recall** (10K) | **✅** | **✅** | **✅** |
| Long-context recall (30K) | n/a (20K cap) | **✅** | **✅** |
| Long-context recall (60K) | n/a | **✅** | **✅** |
| Long-context recall (90K) | n/a | n/a (75K cap) | **✅** |
| Short-prompt TPS (narrative) | 65.9 | 65.2 | **33.0** (cudagraph-off cost) |
| Peak TPS | 85 | 85 | 33 |
| Max context | 20K | **75K** | **125K** |
| Vision | ✅ | ❌ | ✅ |
| VRAM | 22.8 GB | 22.2 GB | 22.0 GB |

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

### Short-prompt TPS stuck at ~30

If you're on `docker-compose.longctx-experimental.yml`, this is **expected** — it ships `--compilation-config '{"cudagraph_mode":"NONE"}'` as a workaround for [#40831](https://github.com/vllm-project/vllm/issues/40831). Sustained ~33 TPS at 125K ctx is the cost of correctness on that variant. Use the default `docker-compose.yml` for ~85 TPS at 20K. If you're seeing ~30 TPS on the default, something else is wrong — check that `patch_tolist_cudagraph.py` applied (`docker logs ... | grep tolist_cudagraph_fix`).

### Tool calls return `<tool_call>{...}</tool_call>` as plain text (tool extraction doesn't fire)

Two possible causes; the logs distinguish them:

**Cause A — you removed `cudagraph_mode=NONE` from the long-ctx compose.** The original 125K config (cudagraph on) hits [#40831](https://github.com/vllm-project/vllm/issues/40831) and produces `<tool_call>` loops. The shipped `docker-compose.longctx-experimental.yml` already includes the cudagraph-off workaround; if you stripped that flag for performance, restore it.

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
│   ├── patch_tolist_cudagraph.py               our CUDA graph capture crash fix (#40807)
│   ├── patch_pr40798_workspace.py              research artifact — backports vllm#40798;
│   │                                            does NOT fix #40831 (probe 8); kept for
│   │                                            reproducibility of the negative result
│   └── genesis/                                (gitignored; fetched by setup.sh)
├── compose/
│   ├── docker-compose.yml                      DEFAULT — MTP + fp8 + vision, 20K, ~85 TPS
│   ├── docker-compose.tools-text.yml           text-only, 75K ctx, ~85 TPS
│   └── docker-compose.longctx-experimental.yml 125K + vision via cudagraph-off, ~33 TPS
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
