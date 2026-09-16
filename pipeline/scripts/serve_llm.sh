#!/usr/bin/env bash
# Поднять vLLM с моделью из .env. Запускать в tmux, после transcribe и index.
#
# На кластере ни веса, ни vLLM ставить не нужно: и то, и другое уже лежит в
# общей файловой системе. Окружение подключается только на чтение — как и
# conda-окружение для ASR, мы в него ничего не пишем.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

: "${LLM_MODEL:?задайте LLM_MODEL в .env}"
: "${VLLM_ENV:=/home/jovyan/degainanov/envs/vllm029-cu129}"
: "${VLLM_PORT:=8000}"
: "${VLLM_TP_SIZE:=8}"
: "${VLLM_GPU_MEM_UTIL:=0.85}"
: "${VLLM_MAX_MODEL_LEN:=32768}"
: "${VLLM_MAX_NUM_SEQS:=32}"
: "${VLLM_MAX_BATCHED_TOKENS:=8192}"

if [ ! -x "$VLLM_ENV/bin/vllm" ]; then
  echo "не найден vllm в $VLLM_ENV — задайте VLLM_ENV в .env" >&2
  exit 1
fi

# Локальный путь к весам мы можем проверить заранее; имя с Hugging Face — нет.
case "$LLM_MODEL" in
  /*) [ -d "$LLM_MODEL" ] || { echo "не найдены веса: $LLM_MODEL" >&2; exit 1; } ;;
esac

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "На GPU есть активные процессы. Останови transcribe и index перед vLLM." >&2
  exit 1
fi

export PATH="$VLLM_ENV/bin:$PATH"

# --enable-expert-parallel обязателен для этой модели. Её эксперты имеют
# moe_intermediate_size = 640, а блок FP8-квантования равен 128: при делении
# эксперта между восемью картами выходит кусок в 80 элементов, и vLLM падает
# на "not divisible by weight quantization block_n". Экспертный параллелизм
# раскладывает экспертов по рангам целиком и эту размерность не режет.
exec "$VLLM_ENV/bin/vllm" serve "$LLM_MODEL" \
  --served-model-name "$LLM_MODEL" \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --enable-expert-parallel \
  --moe-backend triton \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$VLLM_MAX_BATCHED_TOKENS"
