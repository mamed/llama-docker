#!/bin/sh
# llama-locked.sh — Hardened launcher for llama-server.
# All flags are baked into the image. Any arguments passed to the container
# at runtime are ignored with a warning. Only the model is configurable
# via the baked CMD.

# Refuse any runtime overrides
if [ "$#" -gt 0 ]; then
    echo "WARNING: llama-locked.sh ignores runtime args: $*" >&2
    echo "All configuration is baked into the image. See Dockerfile for details." >&2
fi

exec /usr/local/bin/llama-server \
    -m /models/unsloth/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 \
    --port 64007 \
    -ngl 99 \
    --ctx-size 262144 \
    -ctk q4_0 \
    -ctv q4_0 \
    --parallel 1 \
    --cont-batching \
    -fa on \
    --override-tensor "blk\.[9]\.ffn_.*_exps=CPU,blk\.1[0-9]\.ffn_.*_exps=CPU,blk\.[2-3][0-9]\.ffn_.*_exps=CPU,blk\.4[0-7]\.ffn_.*_exps=CPU" \
    --chat-template-file /usr/local/share/llama-server/qwen3-coder-native.jinja \
    --reasoning-budget -1 \
    --agent
