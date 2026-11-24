#!/bin/sh
# llama-locked.7b.sh — Hardened launcher for llama-qwen-7b-coder.
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.7b" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/unsloth/Qwen2.5-Coder-7B-Instruct-Q8_0/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf \
    --host 0.0.0.0 --port 64004 \
    -ngl 99 --ctx-size 131072 \
    -ctk q8_0 -ctv q8_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --jinja \
    --reasoning-budget -1 \
    --agent
