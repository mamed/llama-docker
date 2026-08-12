#!/bin/sh
# llama-locked.14b.sh — Hardened launcher for llama-qwen-14b-coder.
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.14b" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/unsloth/Qwen2.5-Coder-14B-Instruct-Q4_K_M/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 64005 \
    -ngl 99 --ctx-size 49152 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --reasoning-budget -1 \
    --agent
