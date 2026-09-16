#!/usr/bin/env bash
# Подготовка окружения на GPU-сервере.
#
# Создаёт pipeline/.venv поверх общего conda-интерпретатора.
# Само conda-окружение НЕ меняется: torch, transformers и huggingface-hub
# читаются оттуда через --system-site-packages. Каталог общего окружения
# доступен на запись, поэтому перед каждой установкой скрипт проверяет,
# что активирован именно наш venv.
#
# На кластере нельзя `uv sync --extra gpu`: extra ставит transformers 5 и
# huggingface-hub 1.x в venv, они перекрывают conda и ломают ASR
# (transformers 4.57 требует huggingface-hub<1).
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

uv pip install -e .
# Эмбеддинги для index; torch/transformers/huggingface-hub — из conda.
uv pip install --no-deps sentence-transformers

# Прошлый sync мог затащить эти пакеты в venv. Тогда они перекрывают conda.
uv pip uninstall -y huggingface-hub tokenizers transformers torch accelerate 2>/dev/null || true

python - <<'PY'
from pathlib import Path

import huggingface_hub
import sentence_transformers
import torch
import transformers

venv = Path(".venv").resolve()


def origin(mod) -> Path:
    return Path(mod.__file__).resolve()


for mod in (torch, transformers, huggingface_hub):
    path = origin(mod)
    if venv in path.parents:
        raise SystemExit(f"{mod.__name__} берётся из venv ({path}), должен быть из conda")

print(f"sentence-transformers {sentence_transformers.__version__}")
print(
    f"torch {torch.__version__} | transformers {transformers.__version__} "
    f"| huggingface_hub {huggingface_hub.__version__} (из общего окружения)"
)
print(f"CUDA-устройств: {torch.cuda.device_count()}")
PY
