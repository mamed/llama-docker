#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Gemma 4 26B-A4B MoE — Max Capable Config + Maximum Thinking
#  Hardware: RTX 5070 Ti 12GB + 9955HX3D + 64GB DDR5
#  Speed: 35-38 tok/s | Context: 262K | Thinking: MAX
# ═══════════════════════════════════════════════════════════════

MODEL_DIR="/run/media/fekry/OS shared/LLM-Models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
MODEL="${MODEL_DIR}/gemma-4-26B_q4_0-it.gguf"
MMPROJ="${MODEL_DIR}/gemma-4-26B-it-mmproj.gguf"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Gemma 4 26B-A4B MoE + Maximum Thinking             ║"
echo "║  18 GPU | KV Q4 | 262K ctx | MM ON | P2 | Think MAX ║"
echo "╚══════════════════════════════════════════════════════╝"

exec llama-server \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --host 0.0.0.0 \
  --port 64000 \
  -ngl 18 \
  --ctx-size 262144 \
  -ctk q4_0 \
  -ctv q4_0 \
  --parallel 2 \
  --cont-batching \
  -fa on \
  --reasoning on \
  --reasoning-budget -1
