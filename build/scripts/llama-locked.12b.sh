#!/bin/sh
# llama-locked.12b.sh — Hardened launcher for llama-qwen-12b image.
# All flags baked in. Runtime overrides ignored.
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.12b" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/google/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf \
    --mmproj /models/google/gemma-4-12B-it-qat-q4_0-gguf/mmproj-gemma-4-12b-it-qat-q4_0.gguf \
    --host 0.0.0.0 --port 64001 \
    -ngl 99 --ctx-size 262144 \
    -ctk q8_0 -ctv q8_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --jinja \
    --reasoning-budget -1 \
    --agent
