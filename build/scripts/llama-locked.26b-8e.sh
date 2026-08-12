#!/bin/sh
# llama-locked.26b-8e.sh — Hardened launcher for llama-26b-8e (multimodal 18 GPU layers).
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.26b-8e" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf \
    --mmproj /models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B-it-mmproj.gguf \
    --host 0.0.0.0 --port 64002 \
    -ngl 18 --ctx-size 237568 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --reasoning-budget -1 \
    --agent
