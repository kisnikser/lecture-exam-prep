"""Step 2 of the answer stage: keep only what a candidate fragment really says.

Retrieval brings back forty fragments per question; most of them merely share
vocabulary with the ticket. This step reads each one and throws away the rest,
so synthesis works on material instead of on search results.
"""

from __future__ import annotations

import asyncio
import hashlib

import structlog
from pydantic import BaseModel, Field

from examprep.answer.verify import quote_matches
from examprep.config import PROMPTS_DIR, get_settings
from examprep.llm import LLMClient, LLMError, render_prompt
from examprep.retrieve import Candidate, search, with_context
from examprep.schemas import Extract, ExtractReading, Question, RelevantChunk
from examprep.store import save_extract

log = structlog.get_logger()

PROMPT_VERSION = "extract_v1"
MAX_QUOTE_LEN = 300


class ExtractResponse(BaseModel):
    """What the model is asked to return for one fragment."""

    relevant: bool
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    points: list[str] = Field(default_factory=list)
    quote: str = ""
    readings: list[str] = Field(default_factory=list)


def timecode(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def input_hash(question: Question, candidates: list[Candidate], model: str) -> str:
    """Identity of this extraction: the question, the fragments, model, prompt.

    Re-running with the same inputs must be a no-op, and any change to the
    retrieved set has to invalidate the result.
    """

    digest = hashlib.sha256()
    digest.update(question.text.encode("utf-8"))
    for candidate in candidates:
        digest.update(candidate.chunk.chunk_id.encode("utf-8"))
        digest.update(candidate.chunk.text.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    digest.update(PROMPT_VERSION.encode("utf-8"))
    return digest.hexdigest()


async def _judge(
    client: LLMClient,
    template: str,
    slug: str,
    question: Question,
    candidate: Candidate,
) -> tuple[Candidate, ExtractResponse | None]:
    prompt = render_prompt(
        template,
        question=question.text,
        chunk_id=candidate.chunk.chunk_id,
        timecode=timecode(candidate.chunk.start),
        fragment=candidate.chunk.text,
        context=with_context(slug, candidate),
    )
    try:
        answer = await client.complete_model(
            prompt,
            ExtractResponse,
            temperature=get_settings().llm_temperature_extract,
        )
    except LLMError:
        log.warning("extract.unparsed", chunk_id=candidate.chunk.chunk_id)
        return candidate, None
    return candidate, answer


async def extract_question(
    client: LLMClient,
    slug: str,
    question: Question,
    top_k: int | None = None,
) -> Extract:
    candidates = search(slug, question.text, top_k=top_k)
    template = (PROMPTS_DIR / f"{PROMPT_VERSION}.md").read_text(encoding="utf-8")

    results = await asyncio.gather(
        *(_judge(client, template, slug, question, candidate) for candidate in candidates)
    )

    relevant: list[RelevantChunk] = []
    readings: list[ExtractReading] = []
    for candidate, answer in results:
        if answer is None or not answer.relevant or not answer.points:
            continue

        quote = answer.quote.strip()[:MAX_QUOTE_LEN]
        if quote and not quote_matches(quote, candidate.chunk.text):
            # The fragment may still be useful; only the quote is unusable.
            log.warning("extract.quote_not_found", chunk_id=candidate.chunk.chunk_id)
            quote = ""

        relevant.append(
            RelevantChunk(
                chunk_id=candidate.chunk.chunk_id,
                relevance=answer.relevance,
                points=answer.points,
                quote=quote or candidate.chunk.text[:MAX_QUOTE_LEN],
            )
        )
        readings.extend(
            ExtractReading(text=item, chunk_id=candidate.chunk.chunk_id)
            for item in answer.readings
            if item.strip()
        )

    relevant.sort(key=lambda item: item.relevance, reverse=True)
    extract = Extract(
        question_id=question.id,
        candidates=len(candidates),
        relevant=relevant,
        readings=readings,
        model=client.model,
        prompt_version=PROMPT_VERSION,
        input_hash=input_hash(question, candidates, client.model),
    )
    save_extract(slug, extract)
    log.info(
        "extract.done",
        question_id=question.id,
        candidates=len(candidates),
        relevant=len(relevant),
    )
    return extract
