"""Build ``data/index.json`` — the manifest the site loads first."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from pydantic import ValidationError

from examprep import store
from examprep.config import DATA_DIR, course_dir
from examprep.schemas import DataIndex, IndexAnswer, IndexCourse

log = structlog.get_logger()

INDEX_PATH = DATA_DIR / "index.json"


def _answers_of(slug: str) -> list[IndexAnswer]:
    answers_dir = course_dir(slug) / "answers"
    if not answers_dir.is_dir():
        return []

    found: list[IndexAnswer] = []
    for path in sorted(answers_dir.glob("*.json")):
        try:
            answer = store.load_answer(slug, path.stem)
        except (ValidationError, ValueError) as exc:
            log.warning("export_index.bad_answer", path=str(path), error=str(exc))
            continue
        found.append(
            IndexAnswer(
                question_id=answer.question_id,
                question_set=answer.question_set,
                coverage=answer.coverage,
                citations=len(answer.citations),
            )
        )
    return found


def export_index() -> DataIndex:
    question_sets = [store.load_question_set(p) for p in store.question_set_paths()]

    crosswalks = []
    for path in store.crosswalk_paths():
        crosswalk = store.load_crosswalk(path)
        crosswalk.id = crosswalk.id or path.stem
        crosswalks.append(crosswalk)

    courses = []
    for path in store.course_paths():
        course = store.load_course(path.parent.name)
        courses.append(
            IndexCourse(
                slug=course.slug,
                title=course.title,
                language=course.language,
                question_sets=course.question_sets,
                videos=len(course.sources),
                answers=_answers_of(course.slug),
            )
        )

    index = DataIndex(
        courses=courses,
        question_sets=question_sets,
        crosswalks=crosswalks,
        generated_at=datetime.now(UTC),
    )
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(index.model_dump_json(indent=2) + "\n", encoding="utf-8")
    log.info("export_index.done", courses=len(courses), question_sets=len(question_sets))
    return index
