#!/usr/bin/env bash
# Switch between Gemma 4 12B (port 64001) and Gemma 4 26B-32E (port 64000).
# They CANNOT run simultaneously on a 12 GiB card — co-residency was attempted
# (see 12b-small profile in docker-compose.yaml) but failed due to VRAM
# fragmentation: when 26B is at ngl 14 (10.3 GiB) the 12B-small at ngl 3
# (162 MiB contiguous needed) couldn't find a clean allocation slot.
#
# This script enforces exclusive operation:
#   ./switch.sh 12b    # stop 26B, start full 12B on port 64001 (~9.3 GiB)
#   ./switch.sh 26b    # stop 12B, start 26B-32E on port 64000 (~11.2 GiB)
#   ./switch.sh status # show current state
#
# After switching, switch the active chat model in Hermes:
#   hermes config set providers.llama-cpp.base_url http://localhost:64000/v1     # 26B
#   hermes config set providers.llama-cpp.base_url http://localhost:64001/v1     # 12B
# (or use provider aliases `llama-cpp` for 26B and `llama-cpp-12b` for 12B)
set -euo pipefail

DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
C12="llama-gemma4-12b"
C26="llama-gemma4-26b-32e"
P12=64001
P26=64000

wait_healthy() {
  local port=$1
  local name=$2
  local label=$3
  echo "Waiting for $label on port $port ..."
  for i in {1..60}; do
    if curl -s -m 2 "http://localhost:${port}/health" 2>/dev/null | grep -q '"ok"'; then
      echo "  ✓ $label healthy"
      return 0
    fi
    sleep 3
  done
  echo "  ✗ $label did NOT become healthy in 180s; check: docker logs ${name}"
  return 1
}

cmd_status() {
  echo "=== Containers ==="
  docker ps -a --filter "name=${C12}" --filter "name=${C26}" \
    --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true
  echo ""
  echo "=== Health ==="
  printf "  port %d (%s): %s\n" "$P26" "26B-32E" "$(curl -s -m 2 http://localhost:${P26}/health 2>/dev/null || echo down)"
  printf "  port %d (%s):     %s\n" "$P12" "12B"     "$(curl -s -m 2 http://localhost:${P12}/health 2>/dev/null || echo down)"
  echo ""
  echo "=== VRAM ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  echo ""
  echo "=== Active config ==="
  echo "  providers.llama-cpp.base_url = $(hermes config get providers.llama-cpp.base_url 2>/dev/null || echo unknown)"
  echo "  providers.llama-cpp-12b.base_url = $(hermes config get providers.llama-cpp-12b.base_url 2>/dev/null || echo unknown)"
  echo "  model (default) = $(hermes config get model 2>/dev/null || echo unknown)"
}

start_12b() {
  echo "[switch] stopping 26B-32E ..."
  docker stop "${C26}" >/dev/null 2>&1 || true
  echo "[switch] starting 12B ..."
  if docker ps -a --format '{{.Names}}' | grep -q "^${C12}$"; then
    docker start "${C12}" >/dev/null 2>&1 || \
      (cd "${DOCKER_DIR}" && docker compose --profile gemma-4-12b up -d 2>&1 | tail -3)
  else
    (cd "${DOCKER_DIR}" && docker compose --profile gemma-4-12b up -d 2>&1 | tail -3)
  fi
  wait_healthy "$P12" "$C12" "12B"
}

start_26b() {
  echo "[switch] stopping 12B ..."
  docker stop "${C12}" >/dev/null 2>&1 || true
  echo "[switch] starting 26B-32E ..."
  if docker ps -a --format '{{.Names}}' | grep -q "^${C26}$"; then
    docker start "${C26}" >/dev/null 2>&1 || \
      (cd "${DOCKER_DIR}" && docker compose --profile gemma-4-26b-32e up -d 2>&1 | tail -3)
  else
    (cd "${DOCKER_DIR}" && docker compose --profile gemma-4-26b-32e up -d 2>&1 | tail -3)
  fi
  wait_healthy "$P26" "$C26" "26B-32E"
}

case "${1:-status}" in
  12b)    start_12b ;;
  26b)    start_26b ;;
  status) cmd_status ;;
  *)      echo "Usage: $0 {12b|26b|status}"; exit 1 ;;
esac
