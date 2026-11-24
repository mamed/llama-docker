#!/bin/sh
# llama-locked.26b-text.sh — Hardened launcher for llama-26b-text.
# Legacy duplicate of llama-26b, kept for backward compat.
# Locked at: ctx 235520 (230*1024), ngl 20, q4_0 KV, reasoning on, port 64003.
# 256K OOMs on 12 GB GPU. 235520 is the user-tuned sweet spot.
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.26b-text" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
    --host 0.0.0.0 --port 64003 \
    -ngl 20 --ctx-size 235520 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --reasoning on --reasoning-budget -1 \
    --agent
