from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from examprep.schemas import Answer, Chunk, Course, QuestionSet


def test_question_set_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate question id"):
        QuestionSet.model_validate(
            {
                "schema_version": 1,
                "id": "demo",
                "title": "Demo",
                "kind": "course",
                "questions": [
                    {"id": "s01", "number": 1, "text": "a"},
                    {"id": "s01", "number": 2, "text": "b"},
                ],
            }
        )


def test_question_set_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        QuestionSet.model_validate(
            {
                "schema_version": 1,
                "id": "demo",
                "title": "Demo",
                "kind": "course",
                "questions": [{"id": "s01", "number": 1, "text": "a"}],
                "typo": 1,
            }
        )


def test_course_rejects_duplicate_video_ids() -> None:
    source = {"video_id": "abc", "url": "https://youtu.be/abc", "title": "t", "order": 1}
    with pytest.raises(ValidationError, match="duplicate video_id"):
        Course.model_validate(
            {
                "schema_version": 1,
                "slug": "demo",
                "title": "Demo",
                "question_sets": ["demo-set"],
                "sources": [source, {**source, "order": 2}],
            }
        )


def test_chunk_id_pattern() -> None:
    assert (
        Chunk.model_validate(
            {"chunk_id": "abc123:0007", "video_id": "abc123", "start": 1.0, "end": 2.0, "text": "x"}
        ).source_type
        == "video"
    )

    with pytest.raises(ValidationError):
        Chunk.model_validate(
            {"chunk_id": "abc123-7", "video_id": "abc123", "start": 1.0, "end": 2.0, "text": "x"}
        )


def _answer(**overrides: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "question_id": "s13",
        "question_set": "skvorchevsky-2025-26",
        "question": "…",
        "answer_md": "…",
        "coverage": "partial",
        "model": "m",
        "prompt_version": "synthesize_v1",
        "input_hash": "h",
        "generated_at": datetime.now(UTC),
        **overrides,
    }


def test_answer_citation_numbers_must_be_dense() -> None:
    citation = {
        "n": 2,
        "chunk_id": "abc123:0007",
        "video_id": "abc123",
        "start": 1.0,
        "quote": "q",
    }
    with pytest.raises(ValidationError, match=r"1\.\.N without gaps"):
        Answer.model_validate(_answer(citations=[citation]))


def test_answer_round_trip() -> None:
    answer = Answer.model_validate(_answer())
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
