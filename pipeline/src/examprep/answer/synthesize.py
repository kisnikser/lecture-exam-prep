"""Step 3 of the answer stage: one spoken-exam answer from the kept fragments."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, Field

from examprep.answer.verify import quote_matches, quote_start
from examprep.config import PROMPTS_DIR, get_settings
from examprep.llm import LLMClient, render_prompt
from examprep.schemas import (
    Answer,
    AnswerReading,
    Chunk,
    Citation,
    Coverage,
    Extract,
    Question,
    QuestionKind,
    RelevantChunk,
    Segment,
)
from examprep.store import save_answer

log = structlog.get_logger()

PROMPT_VERSION = "synthesize_v1"
MAX_QUOTE_LEN = 300

# Сколько фрагментов уходит в синтез. Замер на полном прогоне: билеты с 27-37
# релевантными фрагментами теряли на проверке от половины до почти всех цитат,
# тогда как билеты с дюжиной — единицы. Модель, которой дали сорок тысяч
# символов, перестаёт копировать дословно и начинает пересказывать по памяти.
MATERIAL_LIMIT = 15

# «Лекции покрывают вопрос целиком» — сильное утверждение, и на одной
# подтверждённой цитате оно не держится.
MIN_CITATIONS_FOR_FULL = 3

KIND_NOTE = {
    "general": "общий список МФТИ — тема шире того, что читалось в курсе",
    "course": "список курса — тема разбиралась на лекциях",
}

KIND_GUIDANCE = {
    "general": (
        "Это вопрос из общего списка, и лекции курса покрывают его не полностью. "
        "Держись материала в основной части, а недостающее вынеси в `outside_lectures_md` — "
        "там уместен развёрнутый план ответа по стандартному содержанию темы."
    ),
    "course": (
        "Это вопрос курса, поэтому ориентируйся на акценты самого лектора: его формулировки, "
        "примеры и оценки, а не на учебниковую версию темы. Блок `outside_lectures_md` здесь "
        "нужен редко и должен быть коротким."
    ),
}


class CitationOut(BaseModel):
    chunk_id: str
    quote: str = ""


class ReadingOut(BaseModel):
    text: str
    chunk_id: str


class SynthesisResponse(BaseModel):
    answer_md: str
    citations: list[CitationOut] = Field(default_factory=list)
    coverage: Coverage
    outside_lectures_md: str = ""
    readings: list[ReadingOut] = Field(default_factory=list)


def input_hash(question: Question, extract: Extract, model: str, template: str = "") -> str:
    """Identity of the answer: everything that was actually sent to the model.

    Only the selected fragments enter the hash, not every relevant one, so
    changing how many of them are passed invalidates stored answers.
    """

    digest = hashlib.sha256()
    digest.update(question.text.encode("utf-8"))
    digest.update(extract.input_hash.encode("utf-8"))
    for item in selected(extract):
        digest.update(item.chunk_id.encode("utf-8"))
        digest.update(" ".join(item.points).encode("utf-8"))
    digest.update(model.encode("utf-8"))
    digest.update((template or PROMPT_VERSION).encode("utf-8"))
    return digest.hexdigest()


def selected(extract: Extract) -> list[RelevantChunk]:
    """The fragments synthesis actually gets, most relevant first."""

    return extract.relevant[:MATERIAL_LIMIT]


def format_material(items: list[RelevantChunk], chunks: dict[str, Chunk]) -> str:
    """The kept fragments as the prompt sees them."""

    blocks = []
    for item in items:
        chunk = chunks.get(item.chunk_id)
        if chunk is None:
            continue
        points = "\n".join(f"- {point}" for point in item.points)
        blocks.append(
            f"### {item.chunk_id} ({int(chunk.start) // 60:02d}:{int(chunk.start) % 60:02d})\n"
            f"Тезисы:\n{points}\n\n"
            f"Текст фрагмента:\n```\n{chunk.text}\n```"
        )
    return "\n\n".join(blocks)


def segments_in(chunk: Chunk, segments: Sequence[Segment]) -> list[Segment]:
    """The transcript segments the chunk was built from."""

    return [s for s in segments if s.start < chunk.end and s.end > chunk.start]


def build_citations(
    response: SynthesisResponse,
    chunks: dict[str, Chunk],
    segments_by_video: Mapping[str, Sequence[Segment]] | None = None,
) -> tuple[list[Citation], list[int]]:
    """Turn the model's references into citations, reporting what was dropped.

    A citation survives only if its fragment exists and its quote is really in
    that fragment. Dropped positions come back so the numbering in the answer
    text can be repaired instead of pointing at the wrong fragment.
    """

    citations: list[Citation] = []
    dropped: list[int] = []

    for position, item in enumerate(response.citations, start=1):
        chunk = chunks.get(item.chunk_id)
        quote = item.quote.strip()[:MAX_QUOTE_LEN]
        if chunk is None:
            log.warning("synthesize.unknown_chunk", chunk_id=item.chunk_id)
            dropped.append(position)
            continue
        if not quote or not quote_matches(quote, chunk.text):
            log.warning("synthesize.quote_not_found", chunk_id=item.chunk_id)
            dropped.append(position)
            continue

        # Point at the sentence, not at the top of a ninety-second window.
        start = chunk.start
        if segments_by_video:
            segments = segments_in(chunk, segments_by_video.get(chunk.video_id, []))
            start = quote_start(segments, quote, fallback=chunk.start)

        citations.append(
            Citation(
                n=len(citations) + 1,
                chunk_id=chunk.chunk_id,
                video_id=chunk.video_id,
                start=start,
                quote=quote,
            )
        )

    return citations, dropped


def renumber(answer_md: str, response: SynthesisResponse, dropped: list[int]) -> str:
    """Rewrite ``[n]`` markers after some citations were dropped.

    Without this the text would keep pointing at positions that no longer
    exist, which is worse than having no marker at all.
    """

    if not dropped:
        return answer_md

    mapping: dict[int, int] = {}
    kept = 0
    for position in range(1, len(response.citations) + 1):
        if position in dropped:
            continue
        kept += 1
        mapping[position] = kept

    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        new = mapping.get(old)
        return f"[{new}]" if new else ""

    return re.sub(r"\[(\d+)\]", replace, answer_md)


def verified_coverage(claimed: Coverage, citations: int, question_id: str) -> Coverage:
    """Temper the model's own verdict with how much of it survived checking.

    The model judges coverage before its quotes are verified, so an answer can
    claim the lectures while nothing in it is traceable to them. What the
    student needs to know is whether the answer can be leaned on, and that is
    decided by the citations that held up.
    """

    if citations == 0:
        if claimed != "not_found":
            log.warning("synthesize.no_citations", question_id=question_id, claimed=claimed)
        return "not_found"
    if claimed == "full" and citations < MIN_CITATIONS_FOR_FULL:
        log.warning("synthesize.thin_full", question_id=question_id, citations=citations)
        return "partial"
    return claimed


async def synthesize_question(
    client: LLMClient,
    slug: str,
    question: Question,
    kind: QuestionKind,
    question_set: str,
    extract: Extract,
    chunks: dict[str, Chunk],
    segments_by_video: Mapping[str, Sequence[Segment]] | None = None,
) -> Answer:
    template = (PROMPTS_DIR / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")
    prompt = render_prompt(
        template,
        question_id=question.id,
        question=question.text,
        kind_note=KIND_NOTE[kind],
        kind_guidance=KIND_GUIDANCE[kind],
        material=format_material(selected(extract), chunks)
        or "(лекции ничего не дали по этому вопросу)",
    )

    response = await client.complete_model(
        prompt,
        SynthesisResponse,
        temperature=get_settings().llm_temperature_synthesize,
        max_tokens=8192,
    )

    citations, dropped = build_citations(response, chunks, segments_by_video)
    answer_md = renumber(response.answer_md, response, dropped)

    coverage = verified_coverage(response.coverage, len(citations), question.id)

    readings = [
        AnswerReading(
            text=item.text,
            video_id=chunks[item.chunk_id].video_id,
            start=chunks[item.chunk_id].start,
        )
        for item in response.readings
        if item.chunk_id in chunks and item.text.strip()
    ]

    answer = Answer(
        question_id=question.id,
        question_set=question_set,
        question=question.text,
        answer_md=answer_md,
        outside_lectures_md=response.outside_lectures_md.strip() or None,
        citations=citations,
        readings=readings,
        coverage=coverage,
        model=client.model,
        prompt_version=PROMPT_VERSION,
        input_hash=input_hash(question, extract, client.model, template),
        generated_at=datetime.now(UTC),
    )
    save_answer(slug, answer)
    log.info(
        "synthesize.done",
        question_id=question.id,
        coverage=coverage,
        citations=len(citations),
        dropped=len(dropped),
    )
    return answer
