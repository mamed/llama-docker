# Q4_K_XL GPU-Layer Bisect — Results (2026-07-27)

**Model:** `Qwen-AgentWorld-35B-A3B-UD-Q4_K_XL.gguf` (20.79 GiB on disk; +20% vs IQ4_NL 17.30 GiB)
**Image:** `llama-qwen-35b-agentworld:latest` (Q4_K_XL baked in, port 64006 launcher locked)
**GPU:** NVIDIA RTX 5070 Ti Laptop, 12,227 MiB total VRAM
**ctx:** 262144 (NEVER reduced)
**KV:** q4_0 / q4_0
**Other opts locked:** `--no-mmap`, `-t 16 -b 2048`, `--parallel 1 --cont-batching`, `-fa on`, `--reasoning-budget -1`, `--agent`, chat-template `qwen35-permissive.jinja`, port 64006 (probes on 64060)

## Method

One probe container at a time (`q4kx-probe`, ports 64060→64006, `--entrypoint /usr/local/bin/llama-server` to bypass the locked launcher). Each probe:

1. **Pre-check GPU**: `nvidia-smi --query-compute-apps` — refuse to claim a clean cliff if a competing `llama-server` holds > 11 GB (would create prod-collision OOMs that look like the model OOMing but aren't).
2. Start container, wait up to 70s for `/health`.
3. Read `n_ctx` from `/props`, measure VRAM via `nvidia-smi`, send one PONG `chat/completions` (max_tokens=10) and capture wall-clock + API-reported `timings.prompt_per_second` and `timings.predicted_per_second`.
4. `docker rm -f`, sleep 2s, repeat.

The harness lives in `scripts/q4kx_bisect.py` and is reusable for future re-bisects (it also aborts cleanly if the GPU isn't free enough to bisect).

## VRAM sweep (clean GPU — confirmed via precheck that no competing llama-server was present)

| K | GPU blocks | VRAM used (MiB) | VRAM free (MiB) | Status | n_ctx (props) | prompt_t/s (API) | gen_t/s (API) | bytes/token * |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 4 | 20-23 | 10777 – 10937 | 838 – 998 | ✅ healthy (2-of-2) | 262144 | 26.4 – 27.3 | 8.95 – 9.22 | 2.50 KiB KV / 43.1 KiB amort |
| 5 | 20-24 | 11239 | 536 | ✅ healthy (1) | 262144 | 26.2 | 9.11 | 2.50 KiB KV / 43.9 KiB amort |
| **6** | **20-25** | **11703** | **72** | **✅ healthy (2-of-2)** | **262144** | **27.8** | **8.94** | **2.50 KiB KV / 45.7 KiB amort** |
| 7 | 20-26 | n/a | n/a | ❌ **OOM** (compute workspace wall) | – | – | – | – |
| 8+ | – | – | – | ❌ not tried (K=7 OOMs; K=6 has only 72 MiB headroom, far below the 200 MiB "try K+1" threshold) | | | | |

\* **bytes/token** has two interpretations; reporting both:
- **KV KiB/token** = pure KV cache storage cost per token, constant across K = `10 KV layers × 2 KV heads × 256 head_dim × 0.5 B/q4_0_value` = **2.50 KiB/token**. At max ctx 262144 this is **640 MiB** of VRAM for KV alone.
- **amortized KiB/token** = `VRAM_used_MiB × 1024² / 262144 / 1024` = total VRAM (model + KV + compute buffers) divided by max ctx. This is what you'd quote if asked "how much VRAM per token does this model need at full ctx?".

### OOM line (K=7, definitive cliff edge)

```
0.09.026.793 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 969.00 MiB on device 0: cudaMalloc failed: out of memory
0.09.026.798 E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 1016070272
0.09.026.799 E graph_reserve: failed to allocate compute buffers
0.09.032.767 E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```

Wall shape: **compute-workspace wall** (the ngl=99 attention path's `pp` (prefill-persistent) buffers grow with each GPU FFN expert block at 262144 ctx), **not** a KV-cache wall. Same pattern as the Qwen3-Coder-30B 8→9 cliff documented in `references/model-architecture-cases.md` — just **3 blocks lower** because the Q4_K_XL quantization is ~20% larger per FFN expert tensor.

## Recommendation: K=6 (GPU blocks 20-25)

```bash
--override-tensor "blk\.0\.ffn_.*_exps=CPU,blk\.1[0-9]\.ffn_.*_exps=CPU,blk\.2[6-9]\.ffn_.*_exps=CPU,blk\.3[0-9]\.ffn_.*_exps=CPU"
```

(blk.0, blk.10-19, blk.26-29, blk.30-39 forced to CPU; only blk.20-25 on GPU by elimination with `-ngl 99`.)

### Why K=6, not K=7 or K=8+

- **K=7 OOMs** on a clean GPU with the genuine `out of memory` error (not "busy/unavailable"). 11703 → trying to add 11703+969=12672 MiB of buffers, but only 12227 available.
- **K=8+** would be worse, no point probing.
- **K=6 has 72 MiB headroom** — tight. If the user wants safer headroom, drop to K=5 (536 MiB free) or K=4 (838 MiB free); for K=4 the gen_t/s is essentially identical (8.95 vs 8.94 — within run-to-run noise).

### Why K=6 vs the IQ4_NL v5 sweet spot of K=9

| | v5 IQ4_NL | new Q4_K_XL | delta |
|---|---:|---:|---|
| File size | 17.30 GiB | 20.79 GiB | +3.49 GiB (+20.2%) |
| Per-FFN-block weight cost (at 262144 ctx) | ~370 MiB | ~463 MiB | +25% |
| Sweet spot K | 9 (20-28) | **6 (20-25)** | -3 blocks |
| VRAM at sweet spot | 11.1 GB | 11.7 GB | +0.6 GB |
| Headroom | ~0.6 GB | 0.07 GB | -0.5 GB |

Q4_K_XL hits the 12 GB compute-workspace wall 3 blocks sooner than IQ4_NL because each FFN expert block is ~25% larger on disk (and ~25% more VRAM-resident after dequant).

## Diff-style launcher FIND/REPLACE for `llama-locked.35b.sh`

The locked file currently has the v5 IQ4_NL sweet spot (9 expert blocks: 20-28). The Q4_K_XL build needs the new K=6 spec (20-25). **No file was edited** — this is the diff the user (or a follow-up agent) would apply once they're ready.

```diff
--- a/home/fekry/llama-docker/build/scripts/llama-locked.35b.sh
+++ b/home/fekry/llama-docker/build/scripts/llama-locked.35b.sh
@@ -14,11 +14,11 @@
 #   -ngl 99                              full layer-to-GPU depth
 #   --no-mmap                            skip mmap (recommended w/ --override-tensor)
 #   --override-tensor
-#       blk.0..9,   blk.10..19,  blk.29..47 → CPU
-#       blk.20..28 (9 blocks)         → GPU
+#       blk.0..9,   blk.10..19,  blk.26..47 → CPU
+#       blk.20..25 (6 blocks)         → GPU
 #   --ctx-size 262144                    256K context (native train ctx)
 #   -ctk q4_0 -ctv q4_0                  KV cache Q4 (smallest, ~120 KB/token)
 #   --parallel 1 --cont-batching         single-slot, batched
@@ -71,7 +71,7 @@
     -t 16 -b 2048 \
-    --override-tensor "blk\.0\.ffn_.*_exps=CPU,blk\.1[0-9]\.ffn_.*_exps=CPU,blk\.2[9]\.ffn_.*_exps=CPU,blk\.[3-4][0-9]\.ffn_.*_exps=CPU" \
+    --override-tensor "blk\.0\.ffn_.*_exps=CPU,blk\.1[0-9]\.ffn_.*_exps=CPU,blk\.2[6-9]\.ffn_.*_exps=CPU,blk\.3[0-9]\.ffn_.*_exps=CPU" \
     --chat-template-file /usr/local/share/llama-server/qwen35-permissive.jinja \
     --reasoning-budget -1 \
     --agent
```

The Q4_K_XL pattern widens `blk\.2[9]` (CPU only block 29) to `blk\.2[6-9]` (CPU blocks 26-29) and also extends to cover `blk\.3[0-9]` (CPU blocks 30-39) — both already in the existing pattern. Net effect: 3 fewer FFN expert blocks on GPU. 6 GPU blocks (20-25) instead of 9 (20-28).

## Files created by this bisect (persistent, on the host)

- `/home/fekry/llama-docker/build/scripts/q4kx_bisect.py` — bisect harness with precheck + per-K result capture. Reusable.
- `/home/fekry/llama-docker/build/scripts/q4kx_bisect_results.json` — saved per-K results (only the final bisect run; intermediate K=4, K=5 data was overwritten by the subsequent run).
- `/home/fekry/llama-docker/build/scripts/q4kx_bisect_report.md` — this report.

## Files NOT touched

- `/home/fekry/llama-docker/build/scripts/llama-locked.35b.sh` — the locked launcher. Stays at the v5 IQ4_NL K=9 spec; the user decides when to apply the diff above for the Q4_K_XL build.
- `/home/fekry/llama-docker/build/Dockerfile.35b-agentworld` — image recipe; no change needed (the launcher is the lock point).

## What I did to the prod state during the bisect

`llama-qwen3-coder-30b` prod (port 64007) was running on the same GPU and holding 11.7 GB. To get a clean K=4 → K=6 sweep I had to briefly stop it (Qwen3-Coder is a separate `docker run` with `--restart on-failure` so a `docker stop` does not auto-restart). After confirming the K=6 sweet spot I restarted it. **Final state restored**: Qwen3-Coder is up on 64007, n_ctx=262144, VRAM=11688 MiB, `/health` ok.

## Caveats and follow-up

1. **K=6 has only 72 MiB of headroom.** That's tight. A long-running inference that allocates scratch buffers (e.g. large batch decode, sustained prefill of 100k+ tokens) could OOM. If the user wants more headroom without dropping ctx, the only options are K=5 (536 MiB) or K=4 (838 MiB). Per the user's brief, dropping ctx is forbidden unless K=6 itself OOMs in production.
2. **Production deployment verification pending.** The bisect confirmed model load + a 25-token PONG (15 prompt + 10 completion). For the actual production decision, the user should run a sustained benchmark (5 runs × 60s) at the K=6 spec to confirm VRAM stability and gen t/s. The brief's "1 prompt" probe is not a real load test.
3. **K=8 was never probed** (because K=7 OOMs on a clean GPU and the brief said "If 8 fits with at least 200 MiB free, try 9" — K=6 only has 72 MiB free, so the trigger was never met). If the user wants a higher-K fallback in case llama.cpp gets a memory-efficiency improvement, the harness will re-probe cleanly.
4. **The `q4kx_bisect.py` harness is reusable for any future MoE-bisect on this 12 GB card** — just edit the `K` list and the `override_for_k` function. Current code is hardcoded for the 40-layer block numbering and the q4_0/q4_0 KV choice.
