#!/usr/bin/env bash
# probe-ngl.sh — find max GPU-resident weight footprint for Qwen-AgentWorld-35B-A3B
# Strategy: start with proven prod config (ngl=99 + experts blk.2-47=CPU = ~5.3 GB GPU),
# then progressively move MORE MoE expert blocks to GPU until OOM.
# Caller passes: <ngl> <override_pattern|-> [ctx]
set -euo pipefail

NGL="${1:?usage: probe-ngl.sh <ngl> <override_pattern|-> [ctx]}"
OVERRIDE="${2:--}"   # '-' = no --override-tensor (max experts on GPU)
CTX="${3:-8192}"
PORT="${PORT:-64099}"
NAME="probe-35b-ngl${NGL}-ov$(echo "${OVERRIDE}" | tr -dc '0-9' | head -c 12)"
MODEL_PATH="${MODEL_PATH:-/models/unsloth/Qwen-AgentWorld-35B-A3B-UD-IQ4_NL/Qwen-AgentWorld-35B-A3B-UD-IQ4_NL.gguf}"
IMG="${IMG:-llama-qwen-35b-agentworld:latest}"

echo "═══ probe: ngl=${NGL} ctx=${CTX} override='${OVERRIDE}' ═══"

docker rm -f "${NAME}" 2>/dev/null || true

# Build docker run args
RUN_ARGS=( -d --name "${NAME}" --gpus all -p "${PORT}:${PORT}" \
  -v "/run/media/fekry/OS shared/LLM-Models:/models:ro" \
  --entrypoint /usr/local/bin/llama-server "${IMG}" \
  -m "${MODEL_PATH}" --host 0.0.0.0 --port "${PORT}" \
  -ngl "${NGL}" --ctx-size "${CTX}" \
  -ctk q4_0 -ctv q4_0 --parallel 1 --cont-batching -fa on \
  --reasoning-budget -1 )
if [ "${OVERRIDE}" != "-" ]; then
  RUN_ARGS+=( --override-tensor "${OVERRIDE}" )
fi

docker run "${RUN_ARGS[@]}" >/dev/null

# Wait for health OR exit
START=$(date +%s)
RESULT="timeout"
for i in $(seq 1 120); do
  STATUS=$(docker inspect -f '{{.State.Status}}' "${NAME}" 2>/dev/null || echo "missing")
  if [ "${STATUS}" = "exited" ] || [ "${STATUS}" = "dead" ]; then
    RESULT="oom_or_exit"
    break
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    RESULT="ok"
    break
  fi
  sleep 1
done
ELAPSED=$(($(date +%s) - START))

if [ "${RESULT}" = "ok" ]; then
  VRAM_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  TPERF=$(curl -sf --max-time 60 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"x","messages":[{"role":"user","content":"Reply with one short sentence."}],"max_tokens":60,"stream":false,"temperature":0.0}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); u=d.get('usage',{}); td=d.get('timings',{}) or {}; ct=(td.get('predicted_ms') or 0)/1000; pt=(td.get('prompt_ms') or 0)/1000; gen_tps=(u.get('completion_tokens',0)/ct if ct>0 else 0); pp_tps=(u.get('prompt_tokens',0)/pt if pt>0 else 0); print(f'gen_tok/s={gen_tps:.2f} prompt_tok/s={pp_tps:.2f} ct={u.get(\"completion_tokens\",0)} pt={u.get(\"prompt_tokens\",0)}')" 2>/dev/null || echo "perf_err")
  echo "  → OK  boot=${ELAPSED}s  vram=${VRAM_MIB}MiB  perf: ${TPERF}"
else
  # Pull last log lines to confirm OOM
  ERR=$(docker logs --tail 8 "${NAME}" 2>&1 | grep -E 'out of memory|alloc_tensor_range|failed to allocate|exiting due to' | head -3 | tr '\n' '|')
  echo "  → FAIL  boot=${ELAPSED}s  cause: ${ERR:-(no OOM log)}"
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
sleep 2