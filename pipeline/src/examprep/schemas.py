"""Pydantic models for every artifact the pipeline reads or writes.

This module is the single source of truth for the data format: the TypeScript
types used by ``site/`` are generated from the JSON Schema of these models.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Sequence
from datetime import datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final[Literal[1]] = 1

QuestionKind = Literal["general", "course"]
Coverage = Literal["full", "partial", "not_found"]
SourceType = Literal["video"]
TranscriptSource = Literal["whisper", "captions"]
CrosswalkStatus = Literal["draft", "verified"]

QUESTION_ID_RE = re.compile(r"^[a-z]{1,4}\d{2,3}$")
CHUNK_ID_RE = re.compile(r"^[\w-]+:\d{4,}$")

QuestionId = Annotated[str, Field(pattern=QUESTION_ID_RE.pattern)]
ChunkId = Annotated[str, Field(pattern=CHUNK_ID_RE.pattern)]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$")]
Seconds = Annotated[float, Field(ge=0.0)]


def _reject_duplicates(values: Sequence[Hashable], label: str) -> None:
    duplicates = sorted(str(v) for v, n in Counter(values).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates}")


class Base(BaseModel):
    """Common configuration: reject unknown keys so typos surface early."""

    model_config = ConfigDict(extra="forbid")


class Artifact(Base):
    schema_version: Literal[1] = SCHEMA_VERSION


# --- question sets -------------------------------------------------------


class Question(Base):
    id: QuestionId
    number: int = Field(ge=1)
    text: str = Field(min_length=1)


class QuestionSet(Artifact):
    id: Slug
    title: str
    description: str | None = None
    kind: QuestionKind
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids_and_numbers(self) -> QuestionSet:
        _reject_duplicates([q.id for q in self.questions], "question id")
        _reject_duplicates([q.number for q in self.questions], "question number")
        return self


class CrosswalkPair(Base):
    course: QuestionId
    general: list[QuestionId] = Field(min_length=1)


class Crosswalk(Artifact):
    """Hand-made topic mapping between a course set and the general set.

    Advisory only: it drives the "related questions" links in the UI and never
    takes part in retrieval.
    """

    id: Slug | None = None
    status: CrosswalkStatus
    pairs: list[CrosswalkPair]

    @model_validator(mode="after")
    def _unique_course_ids(self) -> Crosswalk:
        _reject_duplicates([p.course for p in self.pairs], "course question in pairs")
        return self


# --- course --------------------------------------------------------------


class Source(Base):
    video_id: str = Field(min_length=1)
    url: str
    title: str
    duration_s: Seconds | None = None
    order: int = Field(ge=1)


class Course(Artifact):
    slug: Slug
    title: str
    language: str = "ru"
    question_sets: list[Slug] = Field(min_length=1)
    sources: list[Source] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_sources(self) -> Course:
        _reject_duplicates([s.video_id for s in self.sources], "video_id")
        return self


# --- transcripts and chunks ----------------------------------------------


class Segment(Base):
    start: Seconds
    end: Seconds
    text: str

    @model_validator(mode="after")
    def _ordered(self) -> Segment:
        if self.end < self.start:
            raise ValueError(f"segment ends before it starts: {self.start} > {self.end}")
        return self


class Transcript(Artifact):
    video_id: str = Field(min_length=1)
    source: TranscriptSource
    model: str | None = None
    language: str = "ru"
    segments: list[Segment]


class Chunk(Base):
    chunk_id: ChunkId
    source_type: SourceType = "video"
    video_id: str = Field(min_length=1)
    start: Seconds
    end: Seconds
    text: str = Field(min_length=1)


# --- answer stage --------------------------------------------------------


class ExtractReading(Base):
    """A work the lecturer named out loud, with the fragment that mentions it."""

    text: str = Field(min_length=1)
    chunk_id: ChunkId


class RelevantChunk(Base):
    chunk_id: ChunkId
    relevance: float = Field(ge=0.0, le=1.0)
    points: list[str] = Field(min_length=1)
    quote: str = Field(min_length=1)


class Extract(Artifact):
    question_id: QuestionId
    candidates: int = Field(ge=0)
    relevant: list[RelevantChunk] = Field(default_factory=list)
    readings: list[ExtractReading] = Field(default_factory=list)
    model: str
    prompt_version: str
    input_hash: str


class Citation(Base):
    n: int = Field(ge=1)
    chunk_id: ChunkId
    video_id: str = Field(min_length=1)
    start: Seconds
    quote: str = Field(min_length=1)


class AnswerReading(Base):
    text: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    start: Seconds


class Answer(Artifact):
    question_id: QuestionId
    question_set: Slug
    question: str
    answer_md: str
    outside_lectures_md: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    readings: list[AnswerReading] = Field(default_factory=list)
    coverage: Coverage
    model: str
    prompt_version: str
    input_hash: str
    generated_at: datetime
    error: str | None = None

    @model_validator(mode="after")
    def _citation_numbers_are_dense(self) -> Answer:
        numbers = sorted(c.n for c in self.citations)
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(f"citation numbers must be 1..N without gaps, got {numbers}")
        return self


# --- site index ----------------------------------------------------------


class IndexAnswer(Base):
    """What the pipeline has already produced for one question of one course."""

    question_id: QuestionId
    question_set: Slug
    coverage: Coverage
    citations: int = Field(ge=0)


class IndexCourse(Base):
    slug: Slug
    title: str
    language: str
    question_sets: list[Slug]
    videos: int = Field(ge=0)
    answers: list[IndexAnswer] = Field(default_factory=list)


class DataIndex(Artifact):
    """``data/index.json`` — the only file the site has to fetch first.

    Question sets are inlined: all of them together are a few dozen kilobytes,
    and the browser then needs neither a YAML parser nor a second round trip.
    """

    courses: list[IndexCourse]
    question_sets: list[QuestionSet]
    crosswalks: list[Crosswalk] = Field(default_factory=list)
    generated_at: datetime
