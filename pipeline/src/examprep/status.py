"""A per-stage readiness summary for one course."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import httpx
from pydantic import ValidationError

from examprep import store
from examprep.config import course_dir, get_settings
from examprep.download import audio_path
from examprep.schemas import Coverage


@dataclass
class Stage:
    name: str
    done: int
    total: int
    note: str = ""

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done == self.total


@dataclass
class CourseStatus:
    slug: str
    stages: list[Stage] = field(default_factory=list)
    coverage: dict[str, Counter[Coverage]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    llm: str | None = None


def _count(directory_glob: str, slug: str) -> int:
    path = course_dir(slug)
    return len(list(path.glob(directory_glob)))


def check_llm() -> str:
    """Whether the OpenAI-compatible endpoint is up — checked before `answer`."""

    settings = get_settings()
    url = f"{settings.llm_base_url.rstrip('/')}/models"
    try:
        response = httpx.get(
            url, timeout=3.0, headers={"Authorization": f"Bearer {settings.llm_api_key}"}
        )
    except httpx.HTTPError as exc:
        return f"недоступен ({type(exc).__name__})"
    if response.status_code != 200:
        return f"отвечает {response.status_code}"
    names = [item.get("id", "?") for item in response.json().get("data", [])]
    return f"готов: {', '.join(names) or 'моделей нет'}"


def course_status(slug: str, with_llm: bool = False) -> CourseStatus:
    course = store.load_course(slug)
    status = CourseStatus(slug=slug)
    total = len(course.sources)

    audio = sum(1 for s in course.sources if audio_path(slug, s.video_id).exists())
    transcripts = _count("transcripts/*.json", slug)
    chunks_path = course_dir(slug) / "chunks.jsonl"
    chunks = sum(1 for _ in chunks_path.open(encoding="utf-8")) if chunks_path.exists() else 0
    embeddings = (course_dir(slug) / "embeddings.npy").exists()

    questions = [q for set_id in course.question_sets for q in _questions_of(set_id)]
    extracts = _count("extracts/*.json", slug)
    answers = _count("answers/*.json", slug)

    status.stages = [
        Stage("ingest", total, total, "источников в course.json"),
        Stage("download", audio, total, "аудиофайлов"),
        Stage("transcribe", transcripts, total, "транскриптов"),
        Stage("index", chunks, chunks, f"чанков; эмбеддинги: {'есть' if embeddings else 'нет'}"),
        Stage("extract", extracts, len(questions), "извлечений"),
        Stage("answer", answers, len(questions), "ответов"),
    ]

    for question_id in sorted(p.stem for p in (course_dir(slug) / "answers").glob("*.json")):
        try:
            answer = store.load_answer(slug, question_id)
        except (ValidationError, ValueError) as exc:
            status.errors.append(f"answers/{question_id}.json: {exc}")
            continue
        status.coverage.setdefault(answer.question_set, Counter())[answer.coverage] += 1
        if answer.error:
            status.errors.append(f"answers/{question_id}.json: {answer.error}")

    if with_llm:
        status.llm = check_llm()

    return status


def _questions_of(set_id: str) -> list[str]:
    for path in store.question_set_paths():
        if path.stem == set_id:
            return [q.id for q in store.load_question_set(path).questions]
    return []
