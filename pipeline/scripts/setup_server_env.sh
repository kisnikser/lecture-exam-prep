#!/usr/bin/env bash
# Подготовка окружения на GPU-сервере.
#
# Создаёт отдельный venv в pipeline/.venv поверх общего conda-интерпретатора.
# Само conda-окружение НЕ меняется: оно подключается только на чтение через
# --system-site-packages, все пакеты ставятся внутрь .venv. Каталог общего
# окружения доступен на запись, поэтому перед каждой установкой скрипт
# проверяет, что активирован именно наш venv.
set -euo pipefail

BASE_PYTHON="${BASE_PYTHON:-/home/user/conda/envs/kandinsky-cuda13.0/bin/python}"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPELINE_DIR"

[ -x "$BASE_PYTHON" ] || { echo "нет интерпретатора: $BASE_PYTHON" >&2; exit 1; }

uv venv --python "$BASE_PYTHON" --system-site-packages
# shellcheck disable=SC1091
. .venv/bin/activate

if [ "${VIRTUAL_ENV:-}" != "$PIPELINE_DIR/.venv" ]; then
  echo "venv не активирован — установка отменена, чтобы не задеть общее окружение" >&2
  exit 1
fi

uv pip install -e ".[asr]"
# torch и transformers берём из общего окружения, поэтому без зависимостей.
uv pip install --no-deps sentence-transformers

python - <<'PY'
import ctranslate2, faster_whisper, sentence_transformers, torch, transformers

print(f"faster_whisper {faster_whisper.__version__} | ctranslate2 {ctranslate2.__version__}")
print(f"sentence-transformers {sentence_transformers.__version__}")
print(f"torch {torch.__version__} | transformers {transformers.__version__} (из общего окружения)")
print(f"CUDA-устройств: {ctranslate2.get_cuda_device_count()}")
PY
