# Internals — how this stack actually works

This is the deep-dive companion to the [README](../README.md). Read this when you want to understand:

- Why a 27 B-parameter model with vision works on a single 24 GB consumer card at all
- What "Genesis patches" are doing under the hood and which patch fixes which bug
- The 9-probe forensics trail that isolated the upstream bugs we worked around
- Current upstream-fix status (which bugs are closed, which are open)

If you just want to use the stack, the README is enough.

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

## What v7.14 changes (the surgical fix for the silent tool-call cascade)

[Sandermage's Genesis v7.14](https://github.com/Sandermage/genesis-vllm-patches) shipped 2026-04-25 with the **P65** patch root-causing [vllm#40880](https://github.com/vllm-project/vllm/issues/40880):

`TurboQuantAttentionImpl._prefill_attention`'s cudagraph-capture bypass treats spec-decode K+1 verify batches as first-chunk prefill (sets `cu_seqlens_k = cu_seqlens_q`), so the captured kernel ignores cached KV. Drafter and verifier both produce noise from the kernel-without-context path; for tool-call prompts they converge on the same high-bias special token (`<tool_call>`) and cascade.

P65 downgrades `_cudagraph_support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE`. vLLM's compilation auto-detects and forces `cudagraph_mode=PIECEWISE` for spec-decode → eager continuation runs the correct branch. 1-token decode batches still get piecewise capture; only K+1 spec-verify batches go eager.

This is a workaround. The proper fix is a custom multi-query Triton kernel (P67) that handles K+1 query against compressed cached KV under cudagraph capture — designed-but-not-implemented in v7.14.

---

## Forensics trail — the 9-probe bug isolation

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

**The Triton kernels are correct when invoked dynamically. torch.compile inductor output is correct.** What corrupts the output is how the captured CUDA graph handles spec-decode's runtime shapes vs warmup-shape capture for the TurboQuant attention path. The ngram path is fixed upstream; the MTP path is closed by Genesis v7.14 P65 (PIECEWISE downgrade) shipping 2026-04-25.

---

## Upstream status

Tracking the bugs we hit and their resolution paths:

- **[#40069](https://github.com/vllm-project/vllm/issues/40069)** — TurboQuant/HIGGS follow-ups tracker (upstream). Lists "Speculative decoding / Eagle" and "Hybrid attention models" as unchecked.
- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — our CUDA graph `.tolist()` crash; worked around locally via `patch_tolist_cudagraph.py`. Sandermage's [Genesis tree](https://github.com/Sandermage/genesis-vllm-patches) reaches the same end state via pre-allocation (Patches 23 + 44).
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — our TurboQuant × spec-decode output-quality bug. **Closed for the ngram path** via Sander's v7.13 backports of upstream PRs (#40738 GDN state recovery + #36138 + #40783 + #39055) plus the [#40875](https://github.com/vllm-project/vllm/issues/40875) `prompt_lookup_min=8` config trick. **MTP path closed** by Genesis v7.14 P65 (cudagraph PIECEWISE downgrade for spec-decode), shipped 2026-04-25.
- **[#40875](https://github.com/vllm-project/vllm/issues/40875)** — Sander's follow-up identifying that `prompt_lookup_min=2` (default) causes ngram to find spurious matches in chat-template tool definitions. Setting `prompt_lookup_min=8` is a config-only fix, validated on Sander's 35B-A3B and confirmed on our 27B (probe 9 Test C). For ngram users, this + v7.13 backports = working stack with cudagraph ON.
- **[#40880](https://github.com/vllm-project/vllm/issues/40880)** — our MTP-specific follow-up filed at Sander's [explicit handoff](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4319965017): *"we did not test MTP at all in the v7.13 cycle... your data shows that assumption is wrong."* MTP × TurboQuant × cudagraph closed by Genesis v7.14 P65 (PIECEWISE downgrade). Proper fix (P67 custom multi-query kernel) designed-but-not-implemented; v7.14 PIECEWISE is the workaround until that lands.
- **[PR #40798](https://github.com/vllm-project/vllm/pull/40798)** — *hypothesized fix that didn't pan out.* Moves `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf` from per-layer `register_buffer(B=max_num_seqs)` to `WorkspaceManager.get_simultaneous()`. Sander and I both expected this would close the pointer-drift between warmup-shape capture and runtime-shape replay. Probe 8 backported the full PR diff via `patches/patch_pr40798_workspace.py` (research artifact, not shipped) and the bug persisted. Useful negative result documented on the PR thread.
- **Sandermage's [P56](https://github.com/Sandermage/genesis-vllm-patches/blob/main/vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py)** — earlier routing-layer workaround (architecturally equivalent to our Probe 4 patch). Superseded by P65's cudagraph-mode downgrade since P56 only addressed the catastrophic surface, not the underlying cudagraph-capture mismatch.
- Sandermage Genesis: we may contribute `patch_tolist_cudagraph.py` to their unified script. They have offered to extract Patches 23 + 44 to upstream.

Until the proper P67 kernel lands: Genesis v7.14 + PIECEWISE cudagraph (the v714 default in this repo) is the correct fix for MTP × TurboQuant. The cudagraph-off variant in `longctx-experimental.yml` is the older workaround; kept around for reference but no longer the recommended path.
