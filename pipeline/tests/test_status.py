from examprep.status import _questions_of, course_status

SLUG = "hps-skvorchevsky"


def test_questions_of_a_known_set() -> None:
    assert len(_questions_of("mipt-hps-general-2025-26")) == 26
    assert _questions_of("нет такого набора") == []


def test_status_lists_every_stage() -> None:
    report = course_status(SLUG)
    assert [stage.name for stage in report.stages] == [
        "ingest",
        "download",
        "prepare-audio",
        "transcribe",
        "index",
        "extract",
        "answer",
    ]
    assert report.stages[0].total == 21
    assert report.llm is None
