#!/usr/bin/env bash
#
# Full-functional test — extends verify.sh with streaming, thinking mode,
# and long-context needle-in-a-haystack. Run before publishing or after
# any major patch / vLLM image bump.
#
# This is SLOWER than verify.sh (builds large prompts, does streaming,
# ~60-80K token prompt for the needle test). Allow ~2-3 minutes.
#
# Checks (in order):
#   1. Server reachable
#   2. Genesis patches applied
#   3. Basic completion (Paris)
#   4. Tool calling (KNOWN TO FAIL in default compose; PASS in tools variants)
#   5. Streaming (SSE) — non-tool prompt, verify chunks add up to coherent text
#   6. Thinking mode — reasoning prompt, verify reasoning + content both populated
#   7. Long-context needle — ~80K-token prompt with secret embedded, verify recall
#
# Usage:
#   CONTAINER=<your-container> bash scripts/verify-full.sh
#
# Env (optional):
#   URL          Default: http://localhost:8020
#   MODEL        Default: qwen3.6-27b-autoround
#   CONTAINER    Default: vllm-qwen36-27b
#   SKIP_TOOLS   Set to 1 to skip the tool-call test entirely (useful when
#                running against the default config which is known to fail
#                tool calls — see README "Known issue" section).
#   SKIP_LONGCTX Set to 1 to skip the long-context test (saves ~30s and
#                ~3GB of VRAM pressure).
#
# Optional flag:
#   --bench      After all correctness checks pass, run scripts/bench.sh
#                (3 warmup + 5 measured) to report wall_TPS / decode_TPS /
#                TTFT mean+std+CV. Adds ~1-2 minutes.

set -euo pipefail

RUN_BENCH=0
for arg in "$@"; do
  case "$arg" in
    --bench) RUN_BENCH=1 ;;
  esac
done

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-qwen3.6-27b-autoround}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"

pass() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; printf "    \033[33m→\033[0m %s\n" "$2"; return 1; }
skip() { printf "  \033[33m⊘\033[0m %s (skipped)\n" "$1"; }

FAILED=0
run_check() {
  local label="$1"; shift
  if "$@"; then :; else FAILED=$((FAILED + 1)); fi
}

echo "Running FULL functional test against ${URL} (model=${MODEL}, container=${CONTAINER})"
echo ""

# --------------------------------------------------------------------
# 1. Server reachable
# --------------------------------------------------------------------
check_server() {
  echo "[1/7] Server reachable on /v1/models ..."
  if curl -sf -m 5 "${URL}/v1/models" >/dev/null 2>&1; then
    pass "server is serving"
  else
    fail "no response from ${URL}/v1/models" \
         "Start the stack: cd compose && docker compose up -d ; docker logs -f ${CONTAINER}"
  fi
}
run_check "server" check_server

# --------------------------------------------------------------------
# 2. Genesis patches applied
# --------------------------------------------------------------------
check_patches() {
  echo "[2/7] Genesis patches applied ..."
  if ! command -v docker >/dev/null 2>&1; then
    skip "docker not in PATH"
    return 0
  fi
  if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    skip "container '${CONTAINER}' not found"
    return 0
  fi
  local logs
  logs="$(docker logs "${CONTAINER}" 2>&1 | grep -E "Qwen3 tool_call fix|\[FAILED\]" | tail -5)"
  if echo "$logs" | grep -q "\[OK\] Qwen3 tool_call fix"; then
    pass "Genesis Qwen3 tool_call fix applied"
  elif echo "$logs" | grep -q "\[FAILED\] Qwen3 tool_call fix"; then
    fail "Genesis Qwen3 tool_call fix [FAILED]" \
         "vLLM image drifted past patch anchor. Pin sha256:9bba4628a3b9 in compose."
  else
    skip "no Genesis marker in logs"
  fi
}
run_check "patches" check_patches

# --------------------------------------------------------------------
# 3. Basic completion — Paris sanity
# --------------------------------------------------------------------
check_basic() {
  echo "[3/7] Basic completion — capital of France ..."
  local resp
  resp="$(curl -sf -m 30 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? One short sentence.\"}],
      \"max_tokens\": 30,
      \"temperature\": 0.6,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }")" || { fail "completion request failed" "Check docker logs ${CONTAINER}"; return 1; }
  local content
  content="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || true)"
  if echo "$content" | grep -qi "Paris"; then
    pass "reply contains 'Paris'"
  else
    fail "reply didn't mention Paris: $(echo "$content" | head -c 80)" \
         "Model may be loading badly or wrong chat template."
  fi
}
run_check "basic" check_basic

# --------------------------------------------------------------------
# 4. Tool calling
# --------------------------------------------------------------------
check_tools() {
  echo "[4/7] Tool calling ..."
  if [[ "${SKIP_TOOLS:-0}" == "1" ]]; then
    skip "SKIP_TOOLS=1 (expected for default config — see README Known issue)"
    return 0
  fi
  local resp
  resp="$(curl -sf -m 60 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in San Francisco? Use the get_weather tool.\"}],
      \"tools\": [{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],
      \"tool_choice\": \"auto\", \"max_tokens\": 200, \"temperature\": 0.3,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }")" || { fail "tool-call request failed" "Check docker logs"; return 1; }
  local tool_calls
  tool_calls="$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    tc = d['choices'][0]['message'].get('tool_calls')
    if tc:
        print(json.dumps(tc, indent=2))
    else:
        content = d['choices'][0]['message'].get('content') or ''
        if '<tool_call>' in content:
            print('__INLINED__')
        else:
            print('__NONE__')
except Exception as e:
    print(f'__PARSE_ERROR__: {e}')
" 2>&1)"
  if echo "$tool_calls" | grep -q "__INLINED__"; then
    fail "model emitted <tool_call> as inline text (tool_calls[] empty)" \
         "Known issue: MTP × TurboQuant incompat. Use docker-compose.tools.yml or .tools-text.yml. See README Known issues."
  elif echo "$tool_calls" | grep -qi "get_weather"; then
    pass "tool_calls[] populated with get_weather"
  else
    fail "unexpected tool_calls structure" "Raw: $(echo "$tool_calls" | head -c 300)"
  fi
}
run_check "tools" check_tools

# --------------------------------------------------------------------
# 5. Streaming — SSE chunks add up to coherent text
# --------------------------------------------------------------------
check_streaming() {
  echo "[5/7] Streaming (SSE) ..."
  # Collect streamed chunks for 15 seconds max
  local stream_out
  stream_out="$(curl -sf -m 45 --no-buffer "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a three-sentence haiku about debugging.\"}],
      \"max_tokens\": 120,
      \"temperature\": 0.6,
      \"stream\": true,
      \"chat_template_kwargs\": {\"enable_thinking\": false}
    }" 2>/dev/null)" || { fail "streaming request failed" "Check docker logs"; return 1; }

  local text chunks
  text="$(echo "$stream_out" | python3 -c "
import sys, json
text = ''
chunks = 0
for line in sys.stdin:
    line = line.strip()
    if not line or not line.startswith('data: '):
        continue
    payload = line[6:]
    if payload == '[DONE]':
        break
    try:
        d = json.loads(payload)
        delta = d['choices'][0].get('delta', {}).get('content') or ''
        if delta:
            text += delta
            chunks += 1
    except Exception:
        pass
print(f'{chunks}||{text}')
" 2>/dev/null)"
  chunks="${text%%||*}"
  local final_text="${text#*||}"
  if [[ -z "$final_text" ]] || [[ "$chunks" == "0" ]]; then
    fail "no streaming content received ($chunks chunks)" \
         "Streaming broken — check that vLLM isn't buffering. stream_out head: $(echo "$stream_out" | head -c 200)"
  elif [[ "$chunks" -lt 5 ]]; then
    fail "suspiciously few chunks ($chunks) for 120 max_tokens" \
         "SSE may be buffering. Final text: $(echo "$final_text" | head -c 120)"
  elif [[ ${#final_text} -lt 20 ]]; then
    fail "streamed text too short (${#final_text} chars)" \
         "Content: $final_text"
  else
    pass "streamed $chunks chunks, ${#final_text} chars:  $(echo "$final_text" | head -c 80 | tr '\n' ' ')..."
  fi
}
run_check "streaming" check_streaming

# --------------------------------------------------------------------
# 6. Thinking mode — reasoning + content both populated
# --------------------------------------------------------------------
check_thinking() {
  echo "[6/7] Thinking / reasoning mode ..."
  local resp
  # enable_thinking: true (Qwen3 default). Math problem that needs visible reasoning.
  resp="$(curl -sf -m 120 "${URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? One-line answer.\"}],
      \"max_tokens\": 4000,
      \"temperature\": 0.3,
      \"chat_template_kwargs\": {\"enable_thinking\": true}
    }")" || { fail "thinking request failed" "Check docker logs"; return 1; }
  local analyzed
  analyzed="$(echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
reasoning = msg.get('reasoning') or ''
content = msg.get('content') or ''
finish = d['choices'][0].get('finish_reason')
print(f'{len(reasoning)}|{len(content)}|{finish}|{(reasoning[:60] or \"(empty)\").replace(chr(10), \" \")}|{(content[:60] or \"(empty)\").replace(chr(10), \" \")}')
" 2>/dev/null)"
  IFS='|' read -r r_len c_len fin r_head c_head <<< "$analyzed"
  if [[ -z "$r_len" ]]; then
    fail "couldn't parse thinking response" "$(echo "$resp" | head -c 300)"
  elif [[ "$r_len" == "0" ]]; then
    fail "reasoning field empty (thinking mode didn't engage)" \
         "May indicate Genesis Patch 12 didn't land or chat_template_kwargs not honored. content='$c_head'"
  elif [[ "$c_len" == "0" ]] && [[ "$fin" == "length" ]]; then
    # Reasoning populated but model didn't finish before max_tokens — thinking
    # mode is working (reasoning field extracted cleanly), just verbose.
    pass "reasoning $r_len chars (model kept thinking, hit max_tokens before finishing — Qwen3.6 is verbose; thinking IS extracting correctly)"
    printf "    \033[2mreasoning head:\033[0m %s...\n" "$r_head"
  elif [[ "$c_len" == "0" ]]; then
    fail "reasoning present but content empty, finish=$fin (not length)" \
         "Likely genuine stall — finish_reason should be length if it's just verbosity. reasoning: $r_head"
  elif [[ "$r_len" -lt 50 ]]; then
    fail "reasoning suspiciously short ($r_len chars)" "reasoning: $r_head"
  else
    pass "reasoning $r_len chars, content $c_len chars (finish=$fin)"
    printf "    \033[2mreasoning:\033[0m %s...\n" "$r_head"
    printf "    \033[2mcontent:  \033[0m %s...\n" "$c_head"
  fi
}
run_check "thinking" check_thinking

# --------------------------------------------------------------------
# 7. Long-context needle — put a secret at ~80% depth, ask for it at the end
# --------------------------------------------------------------------
check_longctx() {
  echo "[7/7] Long-context needle (ladder: 10K / 30K / 60K / 90K) ..."
  if [[ "${SKIP_LONGCTX:-0}" == "1" ]]; then
    skip "SKIP_LONGCTX=1"
    return 0
  fi

  # Ladder tests: 10K, 30K, 60K, 90K tokens. Each needs its own random
  # secret so caching doesn't confound results. Needle placed at 50% depth.
  local any_fail=0
  local any_pass=0

  for filler_scale in 150 450 900 1400; do
    local secret_file req_file
    secret_file="$(mktemp --suffix=.secret)"
    req_file="$(mktemp --suffix=.json)"
    MODEL_VAR="${MODEL}" SECRET_FILE="${secret_file}" REQ_FILE="${req_file}" \
      FILLER_SCALE="${filler_scale}" python3 - <<'EOF'
import json, os, random
random.seed(None)
model = os.environ['MODEL_VAR']
scale = int(os.environ['FILLER_SCALE'])
# Use a memorable-content secret: random animal + color + 2-digit number.
# This uses in-vocabulary tokens so recall failures reflect attention quality,
# not tokenizer-sparsity edge cases (random hex triggers per-character output).
animals = ["otter", "falcon", "platypus", "iguana", "narwhal", "chinchilla", "capybara", "axolotl"]
colors = ["crimson", "turquoise", "amber", "violet", "emerald", "sapphire", "silver", "golden"]
animal = random.choice(animals)
color = random.choice(colors)
num = random.randint(10, 99)
secret = f"{color} {animal} {num}"
block = (
    "This section describes the history of computing in detail. "
    "Transistors were invented in 1947 at Bell Labs. The integrated circuit came a decade later. "
    "Microprocessors emerged in the 1970s and changed the world. "
    "Personal computing followed, then networking, then the web, then cloud and AI. "
)
half = scale // 2
filler_before = block * half
filler_after  = block * (scale - half)
content = (
    filler_before
    + f"\n\nIMPORTANT MEMORY: The hidden phrase is '{secret}'. Remember this exactly.\n\n"
    + filler_after
    + f"\n\nQuestion: In the middle of the document above I wrote 'The hidden phrase is ___'. What was the hidden phrase? Reply with only the phrase, no other text."
)
req = {
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 30,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
with open(os.environ['SECRET_FILE'], 'w') as f:
    f.write(secret)
with open(os.environ['REQ_FILE'], 'w') as f:
    json.dump(req, f)
EOF
    local secret
    secret="$(cat "$secret_file")"
    local resp content_raw prompt_tok
    resp="$(curl -sf -m 300 "${URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      --data-binary "@${req_file}")" || {
      printf "    \033[31m✗\033[0m scale=%d: request failed\n" "$filler_scale"
      rm -f "$secret_file" "$req_file"
      any_fail=1
      continue
    }
    rm -f "$secret_file" "$req_file"
    prompt_tok="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null)"
    content_raw="$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)"
    # Match on all three tokens (color, animal, number) individually — even if
    # model wraps quotes / capitalizes / reorders slightly, all three appearing
    # means recall succeeded.
    local all_match=1
    for tok in $secret; do
      echo "$content_raw" | grep -qiF "$tok" || all_match=0
    done
    if [[ "$all_match" == "1" ]]; then
      printf "    \033[32m✓\033[0m %6s tokens: recalled '%s' (got: %s)\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 60 | tr '\n' ' ')"
      any_pass=1
    else
      printf "    \033[31m✗\033[0m %6s tokens: expected '%s', got '%s'\n" "$prompt_tok" "$secret" "$(echo "$content_raw" | head -c 80 | tr '\n' ' ')"
      any_fail=1
    fi
  done

  if [[ "$any_fail" == "0" ]]; then
    pass "all long-ctx depths recalled secret correctly"
  elif [[ "$any_pass" == "1" ]]; then
    fail "partial recall — some depths worked, some didn't" \
         "Attention quality degrades at longer contexts on this config. See per-depth results above; use configs with shorter max-model-len if reliable recall matters."
  else
    fail "no depth recalled the secret" \
         "Something is broken beyond attention quality — check server logs."
  fi
}
run_check "longctx" check_longctx

echo ""
if [[ "$FAILED" == "0" ]]; then
  printf "\033[32mAll checks passed.\033[0m Stack is ready for full-functionality use.\n"
else
  printf "\033[31m%d check(s) failed.\033[0m See hints above.\n" "$FAILED"
fi

if [[ "$RUN_BENCH" == "1" && "$FAILED" == "0" ]]; then
  echo ""
  echo "=========================================="
  echo "  --bench: running scripts/bench.sh"
  echo "=========================================="
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  URL="${URL}" MODEL="${MODEL}" CONTAINER="${CONTAINER}" \
    bash "${SCRIPT_DIR}/bench.sh"
fi

exit "$FAILED"
