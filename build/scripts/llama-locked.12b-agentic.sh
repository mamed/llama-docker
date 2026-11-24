#!/bin/sh
# llama-locked.12b-agentic.sh — Hardened launcher for llama-gemma4-12b-agentic.
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.12b-agentic" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2/gemma4-v2-Q4_K_M.gguf \
    --host 0.0.0.0 --port 64008 \
    -ngl 99 --ctx-size 262144 \
    -ctk q8_0 -ctv q8_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --jinja \
    --reasoning-budget -1 \
    --agent
