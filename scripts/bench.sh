#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - 3 warmup + N measured runs (default 5)
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - summary: mean / std / CV for both TPS metrics + mean TTFT
#   - shows MTP SpecDecoding metrics from docker logs at the end
#
# Why two TPS metrics:
#   - wall_TPS  = "user-perceived speed" (includes prefill cost)
#   - decode_TPS = "model decode rate" (excludes prefill)
#   For long prompts the two can differ a lot. For short prompts they
#   converge. Reporting both keeps comparisons honest across configs.
#
# Prereq: stack is running and reports "Application startup complete".
#
# Env vars:
#   URL            Endpoint. Default: http://localhost:8020
#   MODEL          Served model name. Default: qwen3.6-27b-autoround
#   CONTAINER      Container for log scraping. Default: vllm-qwen36-27b
#   RUNS           Measured runs. Default: 5
#   WARMUPS        Warm-up runs. Default: 3
#   MAX_TOKENS     completion length. Default: 1000
#   PROMPT         Override the canonical prompt
#   QUIET          Set to 1 to skip per-run lines (just print summary)
#
# Usage:
#   bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh

set -euo pipefail

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
PROMPT="${PROMPT:-Write a detailed 800-word essay explaining transformer attention.}"
QUIET="${QUIET:-0}"

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

# Bench runner: streaming request, capture TTFT + total wall + tokens.
# Emits a single line: "wall_seconds ttft_seconds completion_tokens"
python3 - "$URL" "$MODEL" "$PROMPT" "$MAX_TOKENS" "$WARMUPS" "$RUNS" "$QUIET" << 'PYEOF'
import json, sys, time, urllib.request, statistics as s

URL, MODEL, PROMPT, MAX_TOKENS, WARMUPS, RUNS, QUIET = sys.argv[1:]
MAX_TOKENS = int(MAX_TOKENS); WARMUPS = int(WARMUPS); RUNS = int(RUNS); QUIET = int(QUIET) == 1

def run_once():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content and ttft is None:
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall  # safety: never first-token = treat as non-decode
    return wall, ttft, completion_tokens

def fmt(label, wall, ttft, toks):
    decode_t = max(wall - ttft, 1e-6)
    wtps = toks / wall if wall > 0 else 0
    dtps = toks / decode_t
    return wtps, dtps, ttft, f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  toks={toks:>4d}  wall_TPS={wtps:6.2f}  decode_TPS={dtps:6.2f}"

print(f"=== warmups ({WARMUPS}) ===")
for i in range(WARMUPS):
    try:
        w, t, k = run_once()
        _, _, _, line = fmt(f"warm-{i+1}", w, t, k)
        if not QUIET:
            print(line)
    except Exception as e:
        print(f"  warm-{i+1}  FAIL: {e}")

print(f"\n=== measured ({RUNS}) ===")
walls, decodes, ttfts = [], [], []
for i in range(RUNS):
    try:
        w, t, k = run_once()
        wtps, dtps, ttft, line = fmt(f"run-{i+1}", w, t, k)
        if not QUIET:
            print(line)
        walls.append(wtps); decodes.append(dtps); ttfts.append(ttft)
    except Exception as e:
        print(f"  run-{i+1}  FAIL: {e}")

if walls:
    def stats(name, xs, unit):
        m = s.mean(xs)
        sd = s.stdev(xs) if len(xs) > 1 else 0
        cv = (sd / m * 100) if m > 0 else 0
        return f"  {name:<14s} mean={m:7.2f}{unit}   std={sd:6.2f}   CV={cv:4.1f}%   min={min(xs):.2f}   max={max(xs):.2f}"
    print(f"\n=== summary (n={len(walls)}, ctx_prompt={len(PROMPT)} chars, max_tokens={MAX_TOKENS}) ===")
    print(stats("wall_TPS",   walls,   ""))
    print(stats("decode_TPS", decodes, ""))
    print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std={s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0:5.0f}ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
PYEOF

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# MTP / spec-decode stats
if command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
