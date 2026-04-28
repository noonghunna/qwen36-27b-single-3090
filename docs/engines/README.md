# Inference engines for Qwen3.6-27B — comparison + quick recipes

This repo's main path is **vLLM** because it has the deepest support for Qwen3-Next features (vision, MTP, TurboQuant, full OpenAI API parity). But the model also runs on **llama.cpp** and **SGLang** with different trade-offs. This page compares the three; per-engine pages have setup instructions.

> 🔁 **Coming from the README's Quick start?** It already shipped you the vLLM path. Skim this comparison to see what the alternatives look like, then pick a per-engine page if you want to try one.

---

## At a glance

| Engine | Status on this stack | Per-stream TPS (1× 3090) | Vision | Tool calls | Spec-decode | OpenAI API parity |
|---|---|---|---|---|---|---|
| **[vLLM](VLLM.md)** ⭐ | **Validated, production-grade** (this repo) | 51-55 narr / 67-70 code | ✅ | ✅ | ✅ MTP n=3 | ✅ Full |
| **[llama.cpp](LLAMA_CPP.md)** | Works via [Luce DFlash fork](https://github.com/luce-spec/llama-cpp-dflash) at server level; mainline llama.cpp Qwen3-Next support still landing | 28-60 (varies by quant + DDTree) | ✅ (via mmproj) | ⚠️ Limited (no auto-tool-choice in server) | ✅ DFlash N=5 in fork | ⚠️ Partial |
| **[SGLang](SGLANG.md)** | **Blocked** by same Marlin pad-sub-tile-n bug (vllm#40361 / sglang equivalent); EAGLE spec-decode separately blocked by GDN/DeltaNet rollback | n/a (untested at this state) | ✅ | ✅ | ⚠️ EAGLE blocked on hybrid | ✅ Full |

---

## Pros / cons matrix

### vLLM ⭐

**Pros:**
- Deepest Qwen3-Next feature support upstream
- TurboQuant 3-bit KV cache (lets us reach 192K+ context on a single 3090)
- MTP speculative decoding works out of the box
- Genesis patch ecosystem (Sandermage's tree fixes many compatibility edges)
- Full OpenAI API parity (chat, vision, tools, streaming, reasoning, structured output)
- Active development — bugs we hit get triaged within days

**Cons:**
- Heavyweight — Docker image is ~9 GB
- Longer cold-start (~2 min for compile + cudagraph capture)
- Sensitive to upstream API drift across nightly versions (we pin to dev205 to avoid this)
- Frontier features sometimes ship with bugs we have to patch around (the whole reason this repo exists)

**When to pick:** Production / serious local work / anything that needs the full feature set.

---

### llama.cpp

**Pros:**
- Lightweight — single binary, ~50 MB
- Fastest cold-start (~30 sec)
- Lowest VRAM overhead (no inference framework taxes)
- GGUF support for many quant formats (Q4_K_M, Q5_K_S, IQ4_XS, etc.)
- Works on AMD + Intel + Apple Silicon (vLLM is NVIDIA-only)
- Active community, lots of distros / wrappers (Ollama, LM Studio, LocalAI, etc.)

**Cons:**
- Qwen3-Next family support is a moving target — needs the right binary build
- Server feature parity behind vLLM (no auto-tool-choice in upstream `server`; need wrapper)
- DFlash spec-decode requires a fork ([Luce's llama-cpp-dflash](https://github.com/luce-spec/llama-cpp-dflash))
- Concurrent serving is single-threaded by default (the server forks per request — sluggish under concurrent load)
- No TurboQuant equivalent → max usable context is much lower (~64K with Q4_K_M on 24 GB)

**When to pick:** Quick experiments, embedded use, non-NVIDIA hardware, when you want simplicity over feature completeness.

---

### SGLang

**Pros:**
- Designed for high-throughput serving — RadixAttention prefix sharing, structured-output-aware scheduling
- Often beats vLLM by 10-30% on multi-tenant throughput when both work
- First-class OpenAI API
- Good support for batched structured output (constraint decoding)

**Cons:**
- **Currently blocked on this stack by the same Marlin pad-sub-tile-n bug we hit on vLLM TP=2.** Same kernel-line fix applies (would need a similar patch on SGLang's side or for them to pick up the upstream fix).
- EAGLE spec-decode (their MTP equivalent) is separately blocked by the DeltaNet/GDN hybrid layer not supporting KV rollback — this is a Qwen3-Next architectural issue, not SGLang-specific.
- Smaller community than vLLM; fewer eyes on Qwen3-Next bugs.

**When to pick:** Production multi-tenant serving on models that work cleanly on it (not yet Qwen3.6-27B-int4-AutoRound — track the unblock list below).

**Watch list to unblock SGLang on this stack:**
- Marlin pad-sub-tile-n landing (we [filed PR #40361 on vLLM](https://github.com/vllm-project/vllm/pull/40361); the same fix applies to SGLang's Marlin call site)
- DeltaNet KV rollback support upstream (vllm#39931 / issue #40124 land would unblock EAGLE on Qwen3-Next family across engines)

---

## How to choose

```
Need full Qwen3.6 features (vision, tools, MTP, OpenAI API) on a single 3090?
  → vLLM (this repo's main path)

Want to try llama.cpp / Ollama / LM Studio?
  → llama.cpp page — quick GGUF recipe + Luce DFlash fork pointer

Looking for SGLang?
  → SGLang page — current blocked status + watch list

Already comfortable with one engine and want to know if you're missing
something on the others?
  → Read all three pages for a 10-minute overview
```

---

## Per-engine pages

- **[VLLM.md](VLLM.md)** — current setup (what this repo ships). Brief recap + tuning levers.
- **[LLAMA_CPP.md](LLAMA_CPP.md)** — quick GGUF recipe, vision via mmproj, Luce DFlash fork pointer for spec-decode, gotchas around server feature parity.
- **[SGLANG.md](SGLANG.md)** — current blocked state, what would unblock, when to revisit. TBD recipe placeholder until either Marlin pad lands upstream or DeltaNet rollback lands.

---

## See also

- [docs/INTERNALS.md](../INTERNALS.md) — why this repo picked vLLM specifically (the 9-probe forensics + upstream tracker)
- [docs/USE_CASES.md](../USE_CASES.md) — workload-specific configs for the vLLM path
- [LEARNINGS.md (parent stack)](https://github.com/noonghunna/qwen36-27b-single-3090/blob/master/docs/INTERNALS.md) — why vLLM, why these patches
