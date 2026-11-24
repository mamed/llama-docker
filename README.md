# llama-docker

Model-specific Docker images + helper scripts for the llama.cpp service.
Pairs with [llama-bare-metal](../llama-bare-metal) (the systemd service)
and [llama-cpp-unified](../llama-cpp-unified) (the model registry).

## Layout

```
.env                       # which Dockerfile to build (gitignored)
.env.example               # template
Dockerfile.cuda             # shared CUDA base image
docker-compose.yaml         # stack: backend + router + monitor
build/
  Dockerfile.7b
  Dockerfile.12b
  Dockerfile.14b
  Dockerfile.26b
  Dockerfile.26b-8e
  Dockerfile.26b-32e
  Dockerfile.26b-text
  Dockerfile.26b-32e-text
  Dockerfile.35b-agentworld
  Dockerfile.laguna
  Dockerfile.qwen3-coder-30b
README.md
BENCHMARK.md                # historical benchmark numbers
launch-gemma4-26b.sh       # helper: launch gemma 4 26B container
probe-ngl.sh                # helper: probe ngl sweet spot
switch.sh                   # helper: switch active model
STATE.txt                   # last-known STATE pointer
```

## Build a model

```bash
# Pick a Dockerfile
export MODEL=Dockerfile.26b-32e
docker build -f build/$MODEL .
```

## Use the bare-metal path instead

The bare-metal systemd service (`llama-bare-metal/`) is the recommended
production path — the Docker images here are kept for the cases where
you need a fully isolated environment (CI, distributed deployments,
multi-tenant serving).
