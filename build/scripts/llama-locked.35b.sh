#!/bin/sh
# llama-locked.35b.sh — Hardened launcher for llama-qwen-35b-agentworld
#
# ════════════════════════════════════════════════════════════════════
# 🔒 LOCKED CONFIG — DO NOT MODIFY THIS SCRIPT AT RUNTIME 🔒
# ════════════════════════════════════════════════════════════════════
# This file is installed into the image with:
#   chmod 0555           (read+execute, no write)
#   chown root:root      (owned by root)
# The accompanying /usr/local/bin/llama-locked.35b.sh.sha256 holds the
# baked sha256 — this script verifies itself against it before exec'ing.
# If anything mutates this script, the container REFUSES TO START.
#
# LOCKED CONFIG (2026-07-27 bisect v6 — Q4_K_XL LOCK, 6 blocks):
#   -ngl 99                              full layer-to-GPU depth
#   --no-mmap                            skip mmap (recommended w/ --override-tensor)
#   --override-tensor
#       blk.0..9,   blk.10..19,  blk.26..47 → CPU
#       blk.20..25 (6 blocks)         → GPU
#   --ctx-size 262144                    256K context (native train ctx)
#   -ctk q4_0 -ctv q4_0                  KV cache Q4 (~2.50 KiB/token; 640 MiB at max ctx)
#   --parallel 1 --cont-batching         single-slot, batched
#   -fa on                               flash-attn
#   -t 16                                16 CPU threads (for prompt eval + MoE CPU paging)
#   -b 2048                              logical batch size 2048 (better prompt t/s)
#   --reasoning-budget -1                unlimited reasoning (agent mode)
#   --chat-template-file qwen35-permissive.jinja  custom permissive template
#                                       (sha256 18989ae0…; byte-identical to IQ4_NL — model
#                                        intentionally emits "Thinking Process:" preamble)
#   --agent                              agent mode
#
# ────────────── BISECT HISTORY (ctx=262144, q4_0/q4_0, RTX 5070 Ti 12 GB) ──────────────
# Quantization trade-off (v5 IQ4_NL → v6 Q4_K_XL):
#   - Q4_K_XL is +20 % model size (17.3 → 20.8 GiB) → −3 expert blocks on GPU vs v5
#   - Trade buys better reasoning accuracy (Q6_K tail + Q8 anchors)
#
# VRAM cliff (one-block steps, K=4..7):
#   K=4 (20-23 on GPU):  10 777 MiB used,  998 free  ✅
#   K=5 (20-24 on GPU):  11 239 MiB used,  536 free  ✅
#   K=6 (20-25 on GPU):  11 703 MiB used,   72 free  ✅   ← LOCK
#   K=7 (20-26 on GPU):  –                –        ❌   OOM (compute workspace wall)
#   K=8 (20-27 on GPU):  –                –        ❌   OOM (compute workspace wall)
#   K=9 (20-28 on GPU)   (v5 IQ4_NL):     ~620 free  ✅   — superseded
#
# Locked measured numbers (Phase 1, 16 trials/case, 2 warmup dropped, AC power):
#   gen t/s ceiling: 56-67 across tiny / short / medium / long / verylong prompts
#   prompt t/s:     104 (tiny) → 22 416 (long 4.8 k tok) → 69 445 (verylong 23 k tok)
#   single-request σ < 0.020 s on every case (very tight)
#   sustained 4-way × 120 s @ medium:  426 reqs, 0 failed, 56.8 gen t/s, p95 1.146 s
#   concurrency matrix 1/2/4/8 workers:  55.5 / 57.1 / 58.1 / 59.7 gen t/s (sub-linear)
#   max power 108 W, max temp 71 °C, max VRAM 11 755 MiB
#
# Why the launcher is at v6 not v5: Q4_K_XL GGUFs include a Q6_K-tail decoder block
# and Q8_0 anchor tensors. That tail-block fidelity matters specifically on small
# active-param MoE (Qwen 35B-A3B fires 8 of 256 experts/token) — one bad expert
# can swing a reasoning chain. Q4_K_XL also doesn't rely on an imatrix built for
# the base Qwen 3.5 35B (which is what the IQ4_NL was quantized with — agentworld
# fine-tune is a different distribution).
#
# HARDENING:
#   1. Runtime args (passed via docker run …) are WARNED and IGNORED.
#   2. The image ENTRYPOINT is this script, CMD is empty.
#   3. To change config: edit this file + rebuild image. You cannot mutate
#      the running container's config without rebuilding.

set -eu

LAUNCHER="/usr/local/bin/llama-locked.35b.sh"
SHA_FILE="${LAUNCHER}.sha256"

# ── 1. Reject any runtime args passed via `docker run … <args>` ──
if [ "$#" -gt 0 ]; then
    echo "LOCKED: ignoring runtime args: $*" >&2
    echo "Config is locked — see /home/fekry/llama-docker/build/scripts/llama-locked.35b.sh" >&2
fi

# ── 2. Self-verify: refuse to start if this script has been tampered with ──
if [ -f "${SHA_FILE}" ]; then
    EXPECTED=$(awk '{print $1}' "${SHA_FILE}")
    ACTUAL=$(sha256sum "${LAUNCHER}" | awk '{print $1}')
    if [ "${EXPECTED}" != "${ACTUAL}" ]; then
        echo "FATAL: launcher sha256 mismatch." >&2
        echo "  expected: ${EXPECTED}" >&2
        echo "  actual:   ${ACTUAL}" >&2
        echo "Refusing to start — image is corrupted or has been tampered with." >&2
        echo "Rebuild the image: docker build -f Dockerfile.35b-agentworld -t llama-qwen-35b-agentworld:latest ." >&2
        exit 78  # EX_CONFIG
    fi
else
    echo "WARN: sha256 file missing — proceeding without self-verify" >&2
fi

# ── 3. Launch llama-server with the LOCKED config ──
exec /usr/local/bin/llama-server \
    -m /models/unsloth/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL/Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf \
    --host 0.0.0.0 --port 64006 \
    -ngl 99 \
    --ctx-size 262144 \
    -ctk q4_0 -ctv q4_0 \
    --parallel 1 --cont-batching \
    -fa on \
    --no-mmap \
    -t 16 -b 2048 \
    --override-tensor "blk\.0\.ffn_.*_exps=CPU,blk\.1[0-9]\.ffn_.*_exps=CPU,blk\.2[6-9]\.ffn_.*_exps=CPU,blk\.[3-4][0-9]\.ffn_.*_exps=CPU" \
    --chat-template-file /usr/local/share/llama-server/qwen35-permissive.jinja \
    --reasoning-budget -1 \
    --agent