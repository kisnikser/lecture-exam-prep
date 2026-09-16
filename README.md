# lecture-exam-prep

Подготовка к кандидатскому экзамену по записям лекций: пайплайн транскрибирует видео,
индексирует тексты и генерирует ответы на билеты **строго по материалам лекций** — с цитатами
и ссылками на точный момент видео. Локальный сайт показывает готовые ответы.

Текущий курс: «История и философия науки» К. А. Скворчевского (МФТИ), 21 лекция, ~30 часов.

## Быстрый старт

```bash
cp .env.example .env          # заполнить GPU_HOST и параметры LLM

# ноутбук: метаданные и аудио
cd pipeline && uv sync
uv run examprep validate
uv run examprep ingest   --course hps-skvorchevsky
uv run examprep download --course hps-skvorchevsky
./scripts/sync_audio.sh hps-skvorchevsky

# GPU-сервер: транскрибация, индекс, ответы
cd pipeline && ./scripts/setup_server_env.sh   # venv поверх conda; не --extra gpu
uv run examprep prepare-audio --course hps-skvorchevsky   # m4a → 16 кГц WAV
uv run examprep transcribe --course hps-skvorchevsky --gpus 0,1,2,3,4,5,6,7
uv run examprep index      --course hps-skvorchevsky
./scripts/serve_llm.sh                 # в tmux
uv run examprep answer     --course hps-skvorchevsky
uv run examprep export-index

# ноутбук: сайт
cd site && pnpm install && pnpm dev
```

## Проверки

```bash
cd pipeline && uv run ruff check . && uv run mypy src && uv run pytest
cd site && pnpm lint && pnpm typecheck && pnpm test
```

TypeScript-типы генерируются из pydantic-моделей и в git не редактируются руками:

```bash
cd pipeline && uv run python scripts/export_schemas.py
cd site && pnpm typegen
```

## Что лежит в репозитории

- `pipeline/` — CLI `examprep` (Python 3.12, uv).
- `site/` — статический сайт (Vite + React + TypeScript), только чтение данных.
- `data/question-sets/` — списки экзаменационных вопросов (входные данные, правятся руками).
- `data/courses/<slug>/` — метаданные курса, транскрипты, чанки, извлечения и ответы.

Аудио, эмбеддинги и `.env` в git не попадают.

## Публикация

Сайт публикуется на GitHub Pages по адресу `https://kisnikser.github.io/lecture-exam-prep/`
воркфлоу `.github/workflows/deploy.yml` — при пуше в `main`, если менялись `site/` или `data/`.
Один раз нужно включить в репозитории: Settings → Pages → Source: GitHub Actions.

В публичную сборку **не попадают** расшифровки лекций, чанки и промежуточные извлечения: наружу
идут только ответы с цитатами и ссылками на оригинальные видео. Локально (`pnpm dev`) доступны
все данные.

## Авторские права

Репозиторий не содержит и не распространяет видео- и аудиозаписи лекций. В git попадают только
транскрипты и сгенерированные ответы; ссылки ведут на оригинальные публичные записи на YouTube.
Материалы предназначены исключительно для личной подготовки к экзамену.
