#!/usr/bin/env bash
# Отправить аудио курса на GPU-сервер.
# Использование: ./scripts/sync_audio.sh <course-slug>
#
# Если на сервере есть rsync — идём через него (докачка, прогресс).
# Если нет (частый случай на jupyter-образах) — переносим tar'ом через ssh,
# передавая только те файлы, которых на сервере ещё нет.
set -euo pipefail

SLUG="${1:?укажите слаг курса, например hps-skvorchevsky}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.env" ] && set -a && . "$REPO_ROOT/.env" && set +a

: "${GPU_HOST:?задайте GPU_HOST в .env}"
: "${GPU_REPO_PATH:=~/lecture-exam-prep}"

LOCAL_DIR="$REPO_ROOT/data/courses/$SLUG/audio"
REMOTE_DIR="$GPU_REPO_PATH/data/courses/$SLUG/audio"

ssh "$GPU_HOST" "mkdir -p '$REMOTE_DIR'"

if ssh "$GPU_HOST" 'command -v rsync >/dev/null 2>&1'; then
  rsync -avh --progress --partial "$LOCAL_DIR/" "$GPU_HOST:$REMOTE_DIR/"
  exit 0
fi

echo "На сервере нет rsync — переношу через tar over ssh."

PENDING="$(mktemp)"
trap 'rm -f "$PENDING"' EXIT

REMOTE_FILES="$(ssh "$GPU_HOST" "ls -1 '$REMOTE_DIR' 2>/dev/null" || true)"
( cd "$LOCAL_DIR" && ls -1 ) | while read -r name; do
  if ! printf '%s\n' "$REMOTE_FILES" | grep -qxF "$name"; then
    printf '%s\n' "$name"
  fi
done > "$PENDING"

count="$(wc -l < "$PENDING" | tr -d ' ')"
if [ "$count" -eq 0 ]; then
  echo "Всё уже на сервере."
  exit 0
fi

echo "Передаю файлов: $count"
tar -C "$LOCAL_DIR" -cf - -T "$PENDING" | ssh "$GPU_HOST" "tar -C '$REMOTE_DIR' -xf -"
echo "Готово."
