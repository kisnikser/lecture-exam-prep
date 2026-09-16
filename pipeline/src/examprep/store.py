"""Reading and writing the artifacts under ``data/``.

Every load goes through a pydantic model, so a malformed file fails here
rather than deep inside a pipeline step.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import yaml

from examprep.config import COURSES_DIR, QUESTION_SETS_DIR, course_dir
from examprep.schemas import Answer, Chunk, Course, Crosswalk, Extract, QuestionSet, Transcript

CROSSWALK_PREFIX = "crosswalk-"


def question_set_paths() -> list[Path]:
    if not QUESTION_SETS_DIR.is_dir():
        return []
    return sorted(
        p for p in QUESTION_SETS_DIR.glob("*.yaml") if not p.name.startswith(CROSSWALK_PREFIX)
    )


def crosswalk_paths() -> list[Path]:
    if not QUESTION_SETS_DIR.is_dir():
        return []
    return sorted(QUESTION_SETS_DIR.glob(f"{CROSSWALK_PREFIX}*.yaml"))


def course_paths() -> list[Path]:
    if not COURSES_DIR.is_dir():
        return []
    return sorted(COURSES_DIR.glob("*/course.json"))


def load_question_set(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_crosswalk(path: Path) -> Crosswalk:
    return Crosswalk.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_course(slug: str) -> Course:
    path = course_dir(slug) / "course.json"
    return Course.model_validate_json(path.read_text(encoding="utf-8"))


def save_course(course: Course) -> Path:
    path = course_dir(course.slug) / "course.json"
    return _write_json(path, course)


def load_transcript(slug: str, video_id: str) -> Transcript:
    path = course_dir(slug) / "transcripts" / f"{video_id}.json"
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


def save_transcript(slug: str, transcript: Transcript) -> Path:
    path = course_dir(slug) / "transcripts" / f"{transcript.video_id}.json"
    return _write_json(path, transcript)


def iter_chunks(slug: str) -> Iterator[Chunk]:
    path = course_dir(slug) / "chunks.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Chunk.model_validate_json(line)


def save_chunks(slug: str, chunks: list[Chunk]) -> Path:
    path = course_dir(slug) / "chunks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(c.model_dump_json() for c in chunks)
    path.write_text(f"{body}\n", encoding="utf-8")
    return path


def load_extract(slug: str, question_id: str) -> Extract:
    path = course_dir(slug) / "extracts" / f"{question_id}.json"
    return Extract.model_validate_json(path.read_text(encoding="utf-8"))


def save_extract(slug: str, extract: Extract) -> Path:
    path = course_dir(slug) / "extracts" / f"{extract.question_id}.json"
    return _write_json(path, extract)


def load_answer(slug: str, question_id: str) -> Answer:
    path = course_dir(slug) / "answers" / f"{question_id}.json"
    return Answer.model_validate_json(path.read_text(encoding="utf-8"))


def save_answer(slug: str, answer: Answer) -> Path:
    path = course_dir(slug) / "answers" / f"{answer.question_id}.json"
    return _write_json(path, answer)


def read_lines(path: Path) -> list[str]:
    """Non-empty, non-comment lines of a plain text input file."""

    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _write_json(path: Path, model: Answer | Course | Extract | Transcript) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(model.model_dump_json())
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
