#!/bin/sh
# llama-locked.26b.sh — Hardened launcher for llama-26b (gemma-4-26B-A4B text).
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.26b" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
    --host 0.0.0.0 --port 64003 \
    -ngl 20 --ctx-size 229376 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --reasoning on --reasoning-budget -1 \
    --agent
