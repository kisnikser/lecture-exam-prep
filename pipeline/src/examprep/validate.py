"""Schema validation for the hand-written inputs under ``data/``."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from examprep import store


@dataclass
class ValidationReport:
    checked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_data() -> ValidationReport:
    report = ValidationReport()
    question_ids: dict[str, str] = {}
    set_ids: set[str] = set()

    for path in store.question_set_paths():
        name = path.name
        try:
            question_set = store.load_question_set(path)
        except (ValidationError, ValueError) as exc:
            report.errors.append(f"{name}: {exc}")
            continue

        report.checked.append(f"{name} ({len(question_set.questions)} вопросов)")
        set_ids.add(question_set.id)
        if question_set.id != path.stem:
            report.warnings.append(f"{name}: id «{question_set.id}» не совпадает с именем файла")
        for question in question_set.questions:
            if question.id in question_ids:
                report.errors.append(
                    f"{name}: id «{question.id}» уже занят набором {question_ids[question.id]}"
                )
            question_ids[question.id] = question_set.id

    for path in store.crosswalk_paths():
        name = path.name
        try:
            crosswalk = store.load_crosswalk(path)
        except (ValidationError, ValueError) as exc:
            report.errors.append(f"{name}: {exc}")
            continue

        report.checked.append(f"{name} ({len(crosswalk.pairs)} пар, status={crosswalk.status})")
        for pair in crosswalk.pairs:
            for question_id in [pair.course, *pair.general]:
                if question_id not in question_ids:
                    report.errors.append(f"{name}: неизвестный вопрос «{question_id}»")

    for path in store.course_paths():
        slug = path.parent.name
        try:
            course = store.load_course(slug)
        except (ValidationError, ValueError) as exc:
            report.errors.append(f"{path.parent.name}/course.json: {exc}")
            continue

        report.checked.append(f"{slug}/course.json ({len(course.sources)} источников)")
        if course.slug != slug:
            report.errors.append(
                f"{slug}/course.json: slug «{course.slug}» не совпадает с каталогом"
            )
        for set_id in course.question_sets:
            if set_id not in set_ids:
                report.errors.append(f"{slug}/course.json: неизвестный набор вопросов «{set_id}»")

    return report
