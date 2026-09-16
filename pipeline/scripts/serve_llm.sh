#!/usr/bin/env bash
# Поднять vLLM с параметрами из .env. Запускать в tmux, после transcribe/index.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

: "${LLM_MODEL:?задайте LLM_MODEL в .env}"
: "${VLLM_PORT:=8000}"
: "${VLLM_TP_SIZE:=8}"
: "${VLLM_GPU_MEM_UTIL:=0.90}"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "На GPU есть активные процессы. Останови transcribe/index перед запуском vLLM." >&2
  exit 1
fi

exec vllm serve "$LLM_MODEL" \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-model-len 32768
