#!/bin/sh
# llama-locked.laguna.sh — Hardened launcher for llama-laguna-xs (Poolside Laguna XS 2.1)
#
# ════════════════════════════════════════════════════════════════════
# 🔒 LOCKED CONFIG — DO NOT MODIFY THIS SCRIPT AT RUNTIME 🔒
# ════════════════════════════════════════════════════════════════════
# Same hardening protocol as the 35b agentworld launcher. See
# /home/fekry/llama-docker/build/scripts/llama-locked.35b.sh for the
# full pattern. This launcher bakes:
#   - model path: Laguna-XS-2.1-Q4_K_M.gguf
#   - arch:       laguna (custom arch, supported by llama.cpp master @ 0324696)
#   - port:       64007
#   - chat-template: GGUF-embedded (the model ships with a permissive jinja)
#   - reasoning-budget: -1 (unlimited)
#
# LOCKED CONFIG (2026-07-27 — Laguna XS 2.1 first probe):
#   -ngl 99                              full layer-to-GPU depth (auto-fit)
#   --ctx-size 262144                    256K context (native train ctx)
#   -ctk q4_0 -ctv q4_0                  KV cache Q4 (~1.25 KiB/token — head_kv 8, key_len 128)
#   --parallel 1 --cont-batching         single-slot, batched
#   -fa on                               flash-attn
#   -t 16                                16 CPU threads (for prompt eval + MoE CPU paging)
#   -b 2048                              logical batch size 2048 (better prompt t/s)
#   --reasoning-budget -1                unlimited reasoning (agent mode)
#
# Model: Poolside Laguna XS 2.1 (256 experts × 2.2B = 563B total / 17.6B active MoE)
# File:  /models/lmstudio-community/Laguna-XS-2.1-GGUF/Laguna-XS-2.1-Q4_K_M.gguf
# Size:  18.88 GiB on disk
#
# HARDENING:
#   1. Runtime args (passed via docker run …) are WARNED and IGNORED.
#   2. Self-verifying sha256 baked next to the script at .sha256.

# Self-verify: refuse to exec if this file was tampered with after build.
EXPECTED_SHA=$(cat "$(dirname "$0")/llama-locked.laguna.sh.sha256" 2>/dev/null | awk '{print $1}')
ACTUAL_SHA=$(sha256sum "$0" | awk '{print $1}')
if [ -n "$EXPECTED_SHA" ] && [ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]; then
    echo "FATAL: launcher sha256 mismatch." >&2
    echo "  expected: $EXPECTED_SHA" >&2
    echo "  actual:   $ACTUAL_SHA" >&2
    exit 78
fi

# Refuse any runtime arg overrides
if [ "$#" -ne 0 ]; then
    echo "WARNING: runtime args ignored — config is locked in image." >&2
fi

# Launch
exec /usr/local/bin/llama-server \
    -m /models/lmstudio-community/Laguna-XS-2.1-GGUF/Laguna-XS-2.1-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 64008 \
    -ngl 18 \
    --ctx-size 262144 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 \
    --cont-batching \
    -fa on \
    --no-mmap \
    -t 16 \
    -b 2048 \
    --reasoning-budget -1
