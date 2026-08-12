#!/bin/sh
# llama-locked.26b-32e.sh — Hardened launcher for llama-26b-32e.
# gemma-4-26B-A4B multimodal, ngl 18, expert_used=32, ctx 232K, q4_0 KV
if [ "$#" -gt 0 ]; then
    echo "WARNING: ignoring runtime args: $*" >&2
    echo "Config is locked — see Dockerfile.26b-32e" >&2
fi
exec /usr/local/bin/llama-server \
    -m /models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf \
    --mmproj /models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/mmproj-gemma-4-26B-A4B-it-QAT-BF16.gguf \
    --override-kv gemma4.expert_used_count=int:32 \
    --host 0.0.0.0 --port 64000 \
    -ngl 17 --ctx-size 262144 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --reasoning-budget -1 \
    --agent
