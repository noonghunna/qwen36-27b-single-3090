# Changelog

Notable changes to this recipe over time. README has the current state; this file is the dated history.

## 2026-04-28 — Prefill-OOM tests + safe v714 default

Triggered by ampersandru's production OOM report ([#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1)) — a Hermes-class agent fetching ~25K tokens of news as a tool reply at 192K context crashed the engine.

**Discovered two distinct activation-memory cliffs** on this hardware:
- **Cliff 1** — TurboQuant attention scratch + tool-response prefill, fires on ≥25K-token tool messages at high `--gpu-memory-utilization`. OOM site: TurboQuant forward (dequant scratch + mid_o/output buffers), ~138 MiB allocate.
- **Cliff 2** — DeltaNet/GLA recurrent state buffer, fires on any single prompt above ~50-60K tokens regardless of mem-util. OOM site: `fla.ops.chunk.chunk_gated_delta_rule_fwd_h.h.new_empty(...)`. NT grows linearly with prompt length; chunked-prefill doesn't help.

**Shipped:**
- `verify-full.sh` extended from 7 → 10 checks: #8 tool-response prefill OOM (multi-turn payload with ~25K-token mock tool message; configurable via `PREFILL_TARGET_CHARS`), #9 output-quality / cascade detection (2K-token completion scanned for `<tool_call>` inline cascade and repetitive degeneracy), #10 MTP acceptance length threshold (asserts mean AL ≥ 2.0 from SpecDecoding metrics).
- `verify-full.sh #7` long-context needle ladder now treats engine HTTP 400 (oversize ctx rejection) as a clean "skipped at this depth" rather than a failure.
- `docker-compose.v714.yml` default lowered to **48K + 0.92** — below both cliffs. All 10 checks pass at this combo.
- v714.yml header documents the full opt-in matrix (64K → 205K) with safe single-prompt + tool-prefill envelopes per tier.
- README has comprehensive Activation-memory caveat with three-layer defense (vLLM `--max-model-len` HTTP 400 rejection + agent-framework truncation + system-prompt limits).

TPS unchanged at the new default: 51 narr / 68 code TPS (CV ~2.3%). Hardware-bound.

## 2026-04-27 — Full-matrix re-bench + substrate unification

Discovered and fixed four real compose drift bugs during a complete re-bench cycle:

- **Image split**: composes had drifted across two different vLLM image pins (`@sha256:9bba4628a3...` = `dev21` for default/tools-text/longctx/eager; `:nightly-100c7b65...` = `dev174` for v714/minimal). All six unified to `vllm/vllm-openai:nightly-07351e0883470724dd5a7e9730ed10e01fc99d08` (= `dev205+g07351e088`, Sandermage's documented reference target).
- **`eager.yml` config drift**: shipped with `gpu-memory-utilization=0.92` and `max-model-len=131072` while [@ampersandru](https://github.com/ampersandru)'s actual measurement was `0.97` + `125000`. As-shipped failed to boot at 131K (KV-OOM). Compose deleted entirely — see "Removed: eager.yml" below.
- **`v714.yml` mount path**: `patch_tolist_cudagraph.py` was mounted from `../patches/genesis/patch_tolist_cudagraph.py` but the file is at `../patches/patch_tolist_cudagraph.py`. Docker silently created an empty directory at the bogus path, breaking the patcher. Fixed.
- **Bench harness regression**: commit `a381086` rewrote `scripts/bench.sh` to add streaming TTFT / CV / decode_TPS instrumentation but silently dropped the original code-prompt arm (3 narrative + 2 code → narrative only). Restored as parallel narrative + code runs in one invocation; all README TPS claims re-measured.
- **Genesis exoneration**: ran an A/B between `default.yml` (with Genesis v7.54) and a fresh `no-genesis-mtp.yml` (identical config minus Genesis). Measured within run-to-run variance — Genesis is performance-neutral on this path, not the cause of any TPS shift vs older claims. Cross-rig confirmed by [u/sudeposutemizligi](https://www.reddit.com/r/LocalLLaMA/) on TP=2 + dev45 + no Genesis (55 narrative / 68 code, same hardware class).

## 2026-04-27 — Removed: docker-compose.eager.yml

`docker-compose.eager.yml` was originally proposed by [@ampersandru](https://github.com/ampersandru) in [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1) as a 125K path that bypasses the cudagraph bug class via `--enforce-eager`. It shipped briefly with a "~52-65 TPS at 125K" claim. Re-bench cycle on dev205 + Genesis v7.54 measured 25.5 narr / 32.3 code — strictly dominated by `longctx-experimental.yml` at the same 125K context (38/50 TPS). `--enforce-eager` disables both cudagraph AND torch.compile, paying a real Python-overhead cost on every forward; `cudagraph_mode=NONE` keeps inductor compilation on and is faster while delivering the same context ceiling and feature set. The compose has been removed in favor of long-ctx as the recommended 125K path.

Eager mode is still reachable via `--compilation-config '{"cudagraph_mode":"NONE"}' --enforce-eager` if you genuinely need the full no-graph escape hatch (e.g., for model debugging or P7 GDN dual-stream), but no shipped variant defaults to it.

## 2026-04-27 — Patch hardening

- `docker-compose.eager.yml` initial commit incorrectly claimed "no Genesis patches needed" while still using `--kv-cache-dtype turboquant_3bit_nc`. Updated to mount Genesis P4. Compose has since been removed entirely (see above). Reported by [@walmis](https://github.com/walmis) in [#5](https://github.com/noonghunna/qwen36-27b-single-3090/issues/5).
- `patches/patch_tolist_cudagraph.py` was silently failing on (a) any non-docker setup (hardcoded `dist-packages` path) and (b) any vLLM nightly past the one we initially tested against (multi-line block anchors fragile against upstream rewording). Fixed in [`c34bbf1`](https://github.com/noonghunna/qwen36-27b-single-3090/commit/c34bbf1) — patcher auto-discovers vLLM via `import vllm` and uses single-line regex anchors. Bug reported by [@3dluvr](https://github.com/3dluvr) in [#1](https://github.com/noonghunna/qwen36-27b-single-3090/issues/1).

## 2026-04-25 — Genesis v7.14 (Sandermage upstream)

Genesis v7.14 shipped with the **P65** patch root-causing [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) — the silent tool-call cascade bug under MTP × TurboQuant × cudagraph. P65 downgrades `_cudagraph_support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE`, forcing PIECEWISE cudagraph for spec-decode. Tool calls populate `tool_calls[]` cleanly; cascade gone.

This shipped as a workaround. The proper fix is a custom multi-query Triton kernel (P67) that handles K+1 query against compressed cached KV under cudagraph capture — designed-but-not-implemented as of v7.14.

See [docs/INTERNALS.md](docs/INTERNALS.md) for the full forensics trail (9-probe bug isolation, P65 mechanics, why a custom kernel would close the gap).

## Earlier

Initial release shipped a `docker-compose.longctx-experimental.yml` at 125K with `cudagraph_mode=NONE` as the long-context option. v7.14 superseded this; `longctx-experimental.yml` is now deprecated but kept in the repo for reference.
