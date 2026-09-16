# CLAUDE.md — Exam Prep from Lecture Videos

## Что это за проект

Инструмент подготовки к экзамену. Вход:
1. плейлист YouTube или список ссылок на записи лекций/семинаров;
2. список экзаменационных билетов/вопросов.

Пайплайн транскрибирует видео, индексирует тексты и генерирует ответы на каждый билет **строго по материалам лекций**, с цитатами и ссылками на точный момент видео. Веб-интерфейс показывает готовые ответы.

## Текущий скоуп (MVP)

- **Один курс:** лекции К. А. Скворчевского «История и философия науки» (slug `hps-skvorchevsky`, русский язык).
- **Два списка вопросов:** общий список МФТИ (26 вопросов) и список курса Скворчевского (21 вопрос). См. раздел «Структура экзамена».
- Пайплайн запускается **из CLI** на ноутбуке и на GPU-сервере (8×H100, доступ по SSH). Никаких GitHub Actions и облачных сервисов.
- Сайт запускается **локально** (`pnpm dev`) и только читает данные.
- Только ответы на билеты. Конспекты лекций, BYOK и запуск пайплайна из UI **вне скоупа**.

**Заложить сразу, реализовать позже:** много курсов на одном сайте (структура `data/courses/<slug>/`, выбор курса в UI) и деплой на публичный GitHub Pages (`base` в Vite, HashRouter).

## Структура экзамена

Кандидатский экзамен по ИФН (МФТИ). В билете три вопроса:
1. из **общего списка** по ИФН для аспирантов МФТИ — `data/question-sets/mipt-hps-general-2025-26.yaml` (`g01`–`g26`);
2. из **списка курса**, который прослушал аспирант (в МФТИ пять курсов: Зотов, Коцюба, Лупандин, Скворчевский, Тихеев) — здесь `data/question-sets/skvorchevsky-2025-26.yaml` (`s01`–`s21`);
3. третий вопрос — формат не уточнён, вне скоупа.

Следствия для архитектуры:
- **Списки вопросов — отдельные сущности**, не часть курса. Общий список переиспользуется всеми курсами. Курс ссылается на наборы через `course.json → question_sets`.
- **Лекции курса — единственный корпус** для обоих списков. Вопросы курса (`kind: course`) ожидались покрытыми почти полностью, но **по факту записей 2022/23 года
  это не так**: в корпусе нет Лакатоса, Фейерабенда, Полани, Тулмина, Лаудана, ван Фраассена и Уэвелла,
  то есть под s15, s17, s18, s19 материала нет вовсе, а s10 покрыт частично. В общем списке по той же
  причине задеты g07 (Гоббс), g08 (Спиноза), g09 (Беркли), g18 и g19. Плотно разобраны Витгенштейн
  (161 упоминание), Венский кружок (43), Куайн (21), фальсификация (8); Кун — вскользь. Общие вопросы (`kind: general`) — частично: по ожиданиям владельца без пары в курсе остаются `g02, g05, g09, g10, g20–g26`. Для них особенно важны честный `coverage` и блок `outside_lectures_md`.
- `data/question-sets/crosswalk-general-skvorchevsky.yaml` — **черновик** соответствий «вопрос курса ↔ общие вопросы» (составлен вручную, `status: draft`). Используется **только в UI** («связанные билеты»). В retrieval не участвует: подмешивание `extracts` парного вопроса сделало бы ответы на общие вопросы зависимыми от порядка запуска и сломало бы идемпотентность по `input_hash`. Гибридный поиск находит те же фрагменты сам. Не считать crosswalk истиной — это только сигнал.
- **Записи других лет** (не MVP, но решение принято): плейлисты других годов добавляются строками в тот же
  `sources.txt` и попадают в **один корпус** — `ingest` отсеивает повторы по `video_id`, `chunk_id` включает
  идентификатор видео. Разносить годы по отдельным курсам не нужно: цель — собрать материал вместе, а не
  считать ответы по каждому году. Понадобится к тому моменту: ограничение разнообразия в retrieve (иначе
  top-K заполнится тремя записями одной темы из разных лет) и пометка года у источника для UI.
- Будущее расширение (не MVP): дополнительные текстовые материалы (учебник, конспекты, PDF) как второй тип источника в корпусе, в первую очередь для общих вопросов без пары. Схема чанков уже допускает это через поле `source_type` (см. ниже). **Решение:** вернуться к этому после этапа 6, по фактическому распределению `coverage`, а не по ожиданиям.

## Окружение на GPU-сервере

Узел — CUDA 13, поэтому всё, собранное под CUDA 12, там не работает: это и определило выбор ASR-бэкенда.
Отдельная беда — состояние узла при длинном аптайме: slab разрастается до сотен гигабайт, память
фрагментируется, и любая крупная аллокация уходит в безуспешную компакцию. Симптом — numpy на 87 млн
отсчётов считается минутами вместо 0.1 с. Лечится перезагрузкой узла; этим же замером проверять узел
перед долгим запуском.

Общее conda-окружение `/home/user/conda/envs/kandinsky-cuda13.0` (Python 3.12, torch, transformers)
**менять нельзя**, хотя каталог и доступен на запись.
Поэтому `scripts/setup_server_env.sh` создаёт `pipeline/.venv` поверх его интерпретатора с `--system-site-packages`: torch и transformers читаются оттуда, пакеты проекта ставятся только внутрь venv.
Любая установка — исключительно при активированном venv; скрипт это проверяет и отказывается работать иначе.

На кластере нельзя `uv sync --extra gpu`: extra ставит transformers 5 и huggingface-hub 1.x в venv, они перекрывают conda и ломают ASR (`transformers 4.57` требует `huggingface-hub<1`).
`sentence-transformers` ставить только с `--no-deps`.
Если в venv уже лежит `huggingface-hub>=1`, удалить его — подхватится 0.36 из conda.

## Топология вычислений

```
Ноутбук                                   GPU-сервер (8×H100, ssh)
────────                                  ──────────────────────────
examprep ingest   (yt-dlp, метаданные)
examprep download (yt-dlp → audio/*.m4a)
        │ rsync audio/ ───────────────▶   examprep transcribe   (transformers whisper-large-v3, 8 воркеров, 1 на GPU)
        │                                 examprep index        (bge-m3 на GPU)
        │                                 vllm serve …          (после освобождения GPU от Whisper)
        │                                 examprep answer       (LLM через OpenAI-compatible API)
        │ ◀──────────── git pull ──────   git commit data/ && git push
pnpm dev (site читает ../data)
```

- Сначала попробовать `download` прямо на сервере. Если YouTube отвечает bot-check / 403, скачивать на ноутбуке (домашний IP) и передавать через `rsync`.
- Whisper и vLLM не должны одновременно занимать одни и те же GPU. Порядок: `transcribe` + `index` → остановка → `vllm serve` → `answer`.
- Транзит данных: `audio/` через rsync (в `.gitignore`), JSON-артефакты через git.

## Ключевые ограничения (не нарушать)

- Фронтенд чисто статический: никаких секретов и серверного кода в `site/`.
- В git **не коммитятся**: аудио, веса моделей, эмбеддинги (`*.npy`), `.env`.
- Пайплайн идемпотентен: каждый шаг пропускает готовое (наличие файла + `input_hash`); `--force` пересчитывает.
- Каждый ответ содержит цитаты с таймкодами и поле `coverage`. Модель не выдаёт общие знания за материал лекций.
- Весь GPU-код устойчив к запуску на машине без GPU (CPU fallback для тестов и отладки на ноутбуке).
- Код, идентификаторы, коммиты — на английском. Контент, промпты и UI — на русском.

## Структура репозитория

```
.
├── .claude/CLAUDE.md
├── .github/workflows/ci.yml       # lint + tests, без GPU
├── README.md
├── .env.example
├── pipeline/
│   ├── pyproject.toml                 # uv; extra [gpu]: torch, transformers, sentence-transformers
│   ├── prompts/
│   │   ├── extract_v1.md
│   │   └── synthesize_v1.md
│   ├── scripts/
│   │   ├── export_schemas.py          # JSON Schema из моделей → site/src/types/
│   │   ├── setup_server_env.sh        # venv на сервере поверх общего conda-окружения
│   │   ├── serve_llm.sh               # vllm serve с параметрами из .env
│   │   └── sync_audio.sh              # rsync audio/ на сервер
│   ├── src/examprep/
│   │   ├── cli.py                     # typer: validate | ingest | download | transcribe | index |
│   │   │                              #        prepare-audio | retrieve | answer | status |
│   │   │                              #        export-index
│   │   ├── config.py                  # pydantic-settings
│   │   ├── schemas.py                 # pydantic-модели всех артефактов (источник истины)
│   │   ├── store.py                   # чтение/запись артефактов data/ через модели
│   │   ├── validate.py                # проверка входных файлов схемой
│   │   ├── export_index.py            # сборка data/index.json
│   │   ├── status.py                  # сводка готовности по этапам
│   │   ├── ingest.py
│   │   ├── download.py
│   │   ├── transcribe/
│   │   │   ├── captions.py            # субтитры YouTube (быстрый черновой режим)
│   │   │   ├── whisper.py
│   │   │   └── multi_gpu.py           # пул процессов, CUDA_VISIBLE_DEVICES на воркер
│   │   ├── clean.py                   # фильтр галлюцинаций/повторов, склейка сегментов
│   │   ├── index.py                   # чанкинг + эмбеддинги + BM25
│   │   ├── retrieve.py                # гибридный поиск
│   │   ├── answer/
│   │   │   ├── extract.py             # шаг 2: LLM-извлечение релевантного из кандидатов
│   │   │   └── synthesize.py          # шаг 3: итоговый ответ
│   │   └── llm.py                     # OpenAI-compatible клиент (vLLM / Anthropic через адаптер)
│   ├── evals/
│   └── tests/
├── site/
│   ├── package.json
│   ├── vite.config.ts                 # отдаёт ../data в dev, копирует в dist при сборке
│   └── src/
│       └── types/                     # schemas.json + сгенерированный data.d.ts
└── data/
    ├── index.json                     # курсы, наборы вопросов целиком, crosswalk, список готовых
    │                                   # ответов с coverage (генерируется `export-index`)
    ├── question-sets/                 # вход: общие для всех курсов списки вопросов
    │   ├── mipt-hps-general-2025-26.yaml
    │   ├── skvorchevsky-2025-26.yaml
    │   └── crosswalk-general-skvorchevsky.yaml   # черновик соответствий
    └── courses/
        └── hps-skvorchevsky/          # лекции К. А. Скворчевского
            ├── course.json            # включает question_sets: [...]
            ├── sources.txt            # вход: ссылки/плейлист, по одной на строку
            ├── glossary.txt           # вход: имена и термины для Whisper
            ├── replacements.txt       # опционально: «неверно -> верно» для clean.py
            ├── audio/                 # .gitignore
            ├── transcripts/<video_id>.json
            ├── chunks.jsonl
            ├── embeddings.npy         # .gitignore
            ├── extracts/<question_id>.json   # промежуточные результаты шага 2 (коммитятся, полезны для отладки)
            └── answers/<question_id>.json
```

## Схемы данных

Источник истины — `pipeline/src/examprep/schemas.py`. TS-типы для `site/` генерируются из JSON Schema (`json-schema-to-typescript`). Все файлы содержат `schema_version`.

**course.json**
```json
{
  "schema_version": 1,
  "slug": "hps-skvorchevsky",
  "title": "История и философия науки (К. А. Скворчевский)",
  "language": "ru",
  "question_sets": ["mipt-hps-general-2025-26", "skvorchevsky-2025-26"],
  "sources": [
    {"video_id": "abc123", "url": "https://youtu.be/abc123", "title": "Лекция 1. Предмет философии науки", "duration_s": 5400, "order": 1}
  ]
}
```

**question-sets/<set_id>.yaml** (уже созданы, редактируются человеком)
```yaml
schema_version: 1
id: skvorchevsky-2025-26
title: "…"
description: "…"       # опционально
kind: course            # general | course
questions:
  - {id: s13, number: 13, text: "Концепция развития науки К. Поппера. Принцип фальсификации. Идея «третьего мира»."}
```
`id` вопроса уникален глобально (префикс `g`/`s`, для новых курсов — свой префикс). Ответы лежат в каталоге курса, потому что один и тот же общий вопрос по разным курсам даёт разные ответы.

**transcripts/<video_id>.json**
```json
{
  "schema_version": 1,
  "video_id": "abc123",
  "source": "whisper",          // whisper | captions
  "model": "large-v3",
  "language": "ru",
  "segments": [{"start": 12.4, "end": 17.9, "text": "..."}]
}
```

**chunks.jsonl**
```json
{"chunk_id": "abc123:0007", "source_type": "video", "video_id": "abc123", "start": 610.0, "end": 842.5, "text": "..."}
```

**extracts/<question_id>.json**
```json
{
  "schema_version": 1,
  "question_id": "s13",
  "candidates": 40,
  "relevant": [{"chunk_id": "abc123:0007", "relevance": 0.9, "points": ["краткий тезис из фрагмента"], "quote": "дословная короткая цитата"}],
  "readings": [{"text": "К. Поппер, «Логика научного открытия»", "chunk_id": "abc123:0007"}],
  "input_hash": "sha256(question + chunk_id и текст каждого кандидата + model + prompt_version)"
}
```

**answers/<question_id>.json**
```json
{
  "schema_version": 1,
  "question_id": "s13",
  "question_set": "skvorchevsky-2025-26",
  "question": "...",
  "answer_md": "Markdown со ссылками [1], [2]…",
  "outside_lectures_md": "Опциональный блок: что стоит добавить из общих знаний (явно помечен в UI)",
  "citations": [{"n": 1, "chunk_id": "abc123:0007", "video_id": "abc123", "start": 610.0, "quote": "..."}],
  "readings": [{"text": "К. Поппер, «Логика научного открытия»", "video_id": "abc123", "start": 610.0}],
  "coverage": "full | partial | not_found",
  "model": "…",
  "prompt_version": "synthesize_v1",
  "input_hash": "sha256(question + extract + model + prompt_version)",
  "generated_at": "2026-09-16T12:00:00Z"
}
```

## Этап ответа (подробно)

1. **Retrieve.** Гибрид: эмбеддинги `BAAI/bge-m3` + BM25 (русская лемматизация, например `pymorphy3`), объединение через RRF, top-K = 40 (`RETRIEVE_TOP_K`). Соседние чанки одной лекции склеиваются.
   Чанкинг (`index.py`): окно `CHUNK_SECONDS` = 90 с с перекрытием `CHUNK_OVERLAP_SECONDS` = 15 с, границы выравниваются по сегментам транскрипта (чанк не рвёт фразу).
2. **Extract.** Для каждого кандидата (или батча из нескольких) LLM возвращает JSON: релевантен ли фрагмент билету, тезисы, короткая дословная цитата. Запросы идут конкурентно (`asyncio`, семафор `LLM_CONCURRENCY`, по умолчанию 64). Нерелевантное отбрасывается.
3. **Synthesize.** По отобранным тезисам — ответ для устного экзамена:
   - краткий тезис (2–3 предложения);
   - основное содержание: позиция/концепция, ключевые понятия, аргументы;
   - критика и связи с другими концепциями (если есть в лекциях);
   - «Что могут спросить дополнительно»;
   - каждое утверждение со ссылкой `[n]`;
   - `outside_lectures_md` — только если `coverage != full`, явно помечено. Для `kind: general` блок может быть развёрнутым (полноценный план ответа по стандартному содержанию темы), потому что лекции курса заведомо покрывают общий список не полностью; для `kind: course` — кратким.
   - `readings` — литература, **названная лектором вслух** по этой теме, с таймкодом упоминания. Собирается на шаге extract; если лектор ничего не называл, поле пустое. Собственных рекомендаций модель не добавляет.
   - Для вопроса курса синтез ориентируется на акценты лектора (его формулировки, примеры, оценки), а не на «учебниковую» версию темы.
4. **Validate.** Pydantic-валидация JSON. Проверка, что все `chunk_id` существуют и что `quote` действительно содержится в тексте чанка (нечёткое сравнение). При ошибке — одна повторная попытка, затем запись с флагом ошибки.

Промпты лежат в `pipeline/prompts/`, версия в имени файла. Новая версия промпта инвалидирует кэш.

## LLM

- Единый клиент `llm.py` поверх OpenAI-compatible API.
- **Ничего ставить и качать не нужно.** На кластере уже есть и веса, и vLLM:
  веса — `/home/jovyan/shares/SR008.fs2/me/models/Qwen3.8-Flash-Next-FP8` (178 ГБ),
  окружение — `/home/jovyan/degainanov/envs/vllm029-cu129` (vLLM 0.29.0, torch 2.13.0+cu129).
  Оба чужие и подключаются только на чтение, как и conda-окружение для ASR.
  Пути задаются через `LLM_MODEL` и `VLLM_ENV`, поднимает всё `scripts/serve_llm.sh`.
- **По умолчанию:** vLLM на GPU-сервере, `LLM_BASE_URL=http://localhost:8000/v1`. Модель задаётся в `LLM_MODEL`. Стартовый кандидат — `Qwen/Qwen3.8-Flash-Next-FP8` (125B всего / 6B активных + 51B n-gram embedding + 4B MTP, ~180 ГБ в FP8, контекст 262k) с `--tensor-parallel-size 8`.
- **Thinking-режим.** Qwen3.8-Flash-Next по умолчанию генерирует `<think>…</think>` перед ответом. Для extract его надо выключать (параметр шаблона чата / `chat_template_kwargs`), иначе 1880 вызовов утонут в рассуждениях; для synthesize можно оставить включённым и сравнить на eval. `llm.py` обязан отрезать блок `<think>` до парсинга JSON, даже когда режим выключен.
- **Альтернатива:** Anthropic API (`LLM_PROVIDER=anthropic`, `LLM_API_KEY`) — если open-weight качество не устроит на eval. Запасной open-weight вариант — `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (~235 ГБ, не thinking).
- Для JSON-ответов использовать structured outputs / guided decoding vLLM (`response_format` с JSON Schema), где доступно.
- `temperature` 0.2–0.3 для synthesize, 0 для extract.

## Транскрибация

- **Бэкенд — `transformers` поверх torch, не faster-whisper.** CTranslate2 (движок faster-whisper) собирается только под CUDA 12 — включая последнюю версию 4.8.2 от 31.08.2026, — а на кластере CUDA 13, и `libcublas.so.12` там нет. Разворачивать рядом CUDA 12 не стали сознательно.
- Модель `openai/whisper-large-v3`, `chunk_length_s=30`, `batch_size=16`, `num_beams=5`, `language="ru"`, `return_timestamps=True`. `torch_dtype` по наличию CUDA: `float16` на GPU, `float32` на CPU.
- **Глоссарий в промпт по умолчанию не подаётся** (`--glossary-prompt` включает). При оконном декодировании
  промпт применяется к каждому 30-секундному окну, и модель распознаёт сам промпт вместо речи: на одной
  лекции так пропало 19% таймлайна, а на месте речи оказались списки имён. Объём текста упал с ~51 тыс.
  символов до ~40 тыс.
- Искажённые имена чинятся **после** распознавания через `replacements.txt` — этот путь не может съесть аудио.
  В `clean.py` есть и фильтр эха глоссария, на случай если промпт всё же включат.
- **VAD больше нет** — он был частью faster-whisper. Отсев галлюцинаций целиком на `clean.py`; если этого мало, подключить silero-vad через torch.hub (он на torch, с CUDA 13 совместим).
- Транскрибация принимает **только подготовленный WAV**: без CTranslate2 в процессе не осталось декодера сжатого аудио, поэтому `prepare-audio` обязателен.
- `multi_gpu.py`: очередь видео и N воркеров (N = число GPU, `TRANSCRIBE_GPUS=0,1,…`); каждый воркер — отдельный процесс со своим `CUDA_VISIBLE_DEVICES`.
- `clean.py`: удаление повторяющихся сегментов и типичных галлюцинаций («Субтитры сделал…», «Продолжение следует…»), нормализация пробелов. Словарь замен — `data/courses/<slug>/replacements.txt`, по строке `неверно -> верно`.
- `captions.py` — быстрый режим для отладки пайплайна (`--mode captions`), не основной.
- **`prepare-audio` обязателен перед `transcribe`.** Whisper декодирует сжатое аудио через PyAV в один
  поток: на 85-минутной лекции это заняло ~20 минут при простаивающих GPU. `ffmpeg` делает то же за
  секунды, поэтому аудио один раз конвертируется в 16 кГц моно WAV, а Whisper читает готовый PCM.
  `transcribe_source()` берёт WAV, если он есть, иначе исходный m4a.
- Идентификатор видео на YouTube может начинаться с дефиса (в этом курсе — `-SSdI8Rsj64`). В Python это
  безразлично, но в шелле такое имя уходит в команду как опция: в скриптах только `./`-префикс или
  абсолютные пути.

## Сайт (MVP)

- Vite + React + TypeScript, HashRouter, `base` настраивается через env (для будущего Pages).
- Данные: `data/` доступна в dev через `vite.config.ts` (`publicDir` или копирование при старте).
- Экраны:
  - выбор курса (даже если курс один);
  - список билетов с разделами, бейджем `coverage` и прогрессом «выучено» (localStorage, с try/catch);
  - страница билета: ответ (react-markdown + remark-gfm), цитаты-сноски со ссылками `https://youtu.be/<id>?t=<sec>`, отдельно помеченный блок «Вне материалов лекций», блок «Литература (по лекциям)» при непустом `readings`;
  - поиск по транскриптам (MiniSearch), результат ведёт на таймкод;
  - два набора вопросов на экране курса (вкладки «Общий список» / «Курс Скворчевского»), связанные билеты из crosswalk;
  - **симулятор экзамена:** случайная пара «вопрос из общего списка + вопрос курса», таймер подготовки, ответ скрыт до нажатия;
  - режим самопроверки: скрыть ответ, показать по клику;
  - печать / сохранение в PDF (print CSS).
- Мобильная вёрстка и тёмная тема.

## Команды

```bash
# --- ноутбук ---
cd pipeline && uv sync
uv run examprep validate                                     # схемы data/question-sets/ и course.json
uv run examprep ingest   --course hps-skvorchevsky          # читает data/courses/hps-skvorchevsky/sources.txt
uv run examprep download --course hps-skvorchevsky
./scripts/sync_audio.sh hps-skvorchevsky                     # rsync (или tar) на сервер, хост из .env: GPU_HOST

# --- GPU-сервер ---
cd pipeline && ./scripts/setup_server_env.sh                 # venv поверх conda; не --extra gpu
uv run examprep prepare-audio --course hps-skvorchevsky      # m4a → 16 кГц WAV, обязательно перед transcribe
uv run examprep transcribe --course hps-skvorchevsky --gpus 0,1,2,3,4,5,6,7
uv run examprep index      --course hps-skvorchevsky
uv run examprep retrieve   --course hps-skvorchevsky --question s13   # отладка: top-K на глаз
./scripts/serve_llm.sh                          # в tmux
uv run examprep answer     --course hps-skvorchevsky [--question s13] [--set skvorchevsky-2025-26] [--force] [--dry-run]
uv run examprep status     --course hps-skvorchevsky         # сводка: что готово на каждом этапе
uv run examprep export-index
git add ../data && git commit -m "data(hps-skvorchevsky): …" && git push

# --- ноутбук ---
git pull
cd site && pnpm install && pnpm dev

# --- проверки ---
cd pipeline && uv run pytest && uv run ruff check . && uv run mypy src
cd site && pnpm lint && pnpm typecheck && pnpm test
```

## Этапы и критерии готовности

1. **Каркас.** Структура репо, `schemas.py`, генерация TS-типов, CI (`ci.yml`: lint + tests, без GPU). Готовые файлы в `data/question-sets/` проходят валидацию схемой.
2. **Ingest + download.** `course.json` по реальному плейлисту; аудио скачано хотя бы в одном из мест (сервер или ноутбук) и сконвертировано в WAV (`prepare-audio`).
3. **Transcribe.** Одна лекция вручную проверена на качество (имена философов, термины); затем весь курс на 8 GPU; `status` показывает 100%.
4. **Index.** Для 3 тестовых билетов top-40 на глаз содержит нужные фрагменты (`examprep retrieve --question …`).
5. **Answer v1.** 5 билетов (минимум 3 из курса и 2 общих, из них 1 без пары в crosswalk) сгенерированы, вручную проверены (цитаты подтверждают утверждения, структура подходит для устного ответа). Итерации промпта.
6. **Answer: все билеты.** 47 ответов (26 общих + 21 курса). Отчёт `status`: распределение `coverage` по наборам, ошибки валидации. Список общих вопросов с `coverage != full` — кандидаты на добавление доп. материалов.
7. **Site MVP.** Все экраны выше работают на данных курса `hps-skvorchevsky`.
8. **Eval.** `pipeline/evals/`: 10–15 размеченных билетов; LLM-as-judge проверяет поддержку утверждений цитатами; отчёт в Markdown; сравнение двух моделей/промптов.
9. **(Позже)** Добавление второго курса без изменений кода.
10. **Деплой.** `deploy.yml` публикует сайт на `https://kisnikser.github.io/lecture-exam-prep/`
    (`SITE_BASE=/lecture-exam-prep/`, HashRouter). В публичную сборку не кладутся расшифровки,
    чанки и извлечения — это чужие лекции, а сайт виднее репозитория; ответы несут цитаты с собой.

## Риски

- **Bot-check YouTube на сервере** — fallback: скачивание на ноутбуке. Не реализовывать иные обходы.
- **Общий список шире курса Скворчевского** — `coverage`, `outside_lectures_md`, явная пометка в UI.
- **Искажение имён и терминов Whisper** — глоссарий, ручная проверка первой лекции, при необходимости словарь замен в `clean.py`. Если large-v3 будет путать имена философов, сравнить на той же лекции с GigaAM-v2, NVIDIA Canary-1b-v2 или Gemma 4 E4B. У всех трёх один и тот же минус: длинные записи и таймкоды придётся резать самим — Gemma 4 принимает максимум 30 с аудио за запрос и таймкодов не даёт вовсе, тогда как весь UI держится на ссылке `https://youtu.be/<id>?t=<sec>`.
- **Нехватка VRAM / конфликт GPU** — строгий порядок этапов; `status` проверяет, что vLLM доступен, до старта `answer`.
- **Невалидный JSON от модели** — guided decoding, ретрай, логирование сырых ответов в `pipeline/.logs/` (в `.gitignore`).
- **Авторские права** — репозиторий публичный: коммитятся только транскрипты и ответы, никакого аудио/видео; дисклеймер в README.

## Конвенции

- Python 3.12, uv, ruff, mypy (strict для `schemas.py`), pytest. Сеть, GPU и LLM в тестах — только моки.
- TypeScript strict, ESLint, Vitest; типы данных только сгенерированные.
- Логи через `rich`/`structlog`, прогресс через `rich.progress`.
- Конфиг через `.env` в корне репозитория (`.env.example` рядом), один на pipeline и скрипты: `GPU_HOST`, `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_CONCURRENCY`, `EMBED_MODEL`, `RETRIEVE_TOP_K`, `CHUNK_SECONDS`, `CHUNK_OVERLAP_SECONDS`, `WHISPER_MODEL`, `TRANSCRIBE_GPUS`, `GPU_REPO_PATH`, `VLLM_ENV`, `VLLM_PORT`, `VLLM_TP_SIZE`, `VLLM_GPU_MEM_UTIL`, `SITE_BASE`.
- Conventional Commits; данные — отдельными коммитами `data(<slug>): …`.
- Перед завершением задачи — линтеры и тесты затронутой части.

## Открытые вопросы (уточнить у владельца)

- Проверка черновика `crosswalk-general-skvorchevsky.yaml` (17 пар) — нужен взгляд владельца.
- Формат третьего вопроса билета.
- Финальный выбор LLM после eval (по умолчанию — vLLM + Qwen3-235B, Anthropic как альтернатива).

Закрыто: дополнительные материалы — вернуться после этапа 6; литература — опциональное поле `readings`, только со слов лектора.
