# Production Benchmark — 2026-07-19 (FINAL)

## Config
- **Model:** gemma-4-26B Q4_0 + mmproj
- **Backend:** CUDA (not Vulkan)
- **GPU layers:** 18
- **Context:** 237568 (max safe for -ngl 18)
- **VRAM:** 11684 / 12227 MiB (95.5%)
- **Port:** 0.0.0.0:64000
- **Container:** llama-gemma4-26b (Docker, restart: on-failure)

## Performance (ctx 237568)
| Metric | Value |
|---|---|
| Generate (short, 5-11 tok) | 48-54 t/s |
| Generate (medium, 253 tok) | 46-48 t/s |
| Generate (long, 800 tok) | 43-45 t/s |
| Generate (max, 1276 tok) | 43 t/s |
| Prompt processing | 80-182 t/s |
| TTFT | 60-165ms |
| Cache hit (multi-turn) | 55-59 t/s |
| Stress stability (5 runs) | 44-46 t/s |
| Temperature | 57-61°C |
| Power | 16-24W |

## Accuracy (ctx 237568)
| Metric | Score |
|---|---|
| Direct prompts (19 questions) | 19/19 (100%) |
| Multi-turn conversation | ✅ correct |
| Complex reasoning (affirming consequent) | ✅ correct |
| Einstein's Riddle | ✅ solvable |
| Instruction following | ✅ respects format |
| Memory/fact recall | ✅ correct |

## Comparison (ctx 237568 vs ctx 229376)
| Metric | ctx 237568 | ctx 229376 |
|---|---|---|
| Accuracy | 19/19 (100%) | 17/19 (89%) |
| Speed | 43-53 t/s | 43-53 t/s |
| VRAM | 11684 MiB | 11647 MiB |
| Context | 237K | 229K |

## Boundary found
- ctx 245760: OOM at startup
- ctx 241664: works short, crashes on long generation
- ctx 237568: works all lengths (520 MiB headroom) ← PRODUCTION
- -ngl 18: max for CUDA on 12GB VRAM
- -ngl 17: fallback if 237568 ctx causes issues (can use 262144 ctx)

## Files
- Compose: /home/fekry/llama-docker/docker-compose.yaml
- Dockerfile: /home/fekry/llama-docker/Dockerfile.cuda
- Benchmark: /home/fekry/llama-docker/BENCHMARK.md
- Image: llama-cuda:latest (5.74 GB)
- Container: llama-gemma4-26b
