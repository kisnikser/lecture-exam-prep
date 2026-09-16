"""The answer stage: retrieve, extract, synthesize, one ticket at a time."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog
from pydantic import ValidationError

from examprep import store
from examprep.answer.extract import extract_question
from examprep.answer.synthesize import synthesize_question
from examprep.config import course_dir
from examprep.llm import LLMClient
from examprep.schemas import Answer, Chunk, Question, QuestionKind

log = structlog.get_logger()

__all__ = ["Ticket", "answer_course", "tickets_of"]


@dataclass
class Ticket:
    question: Question
    kind: QuestionKind
    question_set: str


def tickets_of(
    slug: str, set_id: str | None = None, question_id: str | None = None
) -> list[Ticket]:
    """Every question the course answers, optionally narrowed to one."""

    course = store.load_course(slug)
    wanted = {set_id} if set_id else set(course.question_sets)

    tickets: list[Ticket] = []
    for path in store.question_set_paths():
        question_set = store.load_question_set(path)
        if question_set.id not in wanted or question_set.id not in course.question_sets:
            continue
        for question in question_set.questions:
            if question_id and question.id != question_id:
                continue
            tickets.append(
                Ticket(question=question, kind=question_set.kind, question_set=question_set.id)
            )
    return tickets


def _is_current(slug: str, ticket: Ticket, model: str) -> bool:
    """Whether a stored answer was built from today's inputs.

    Cheap to check and it saves a full extract plus synthesis per ticket, which
    is what makes re-running the stage after a crash bearable.
    """

    try:
        answer = store.load_answer(slug, ticket.question.id)
        extract = store.load_extract(slug, ticket.question.id)
    except (FileNotFoundError, ValidationError, ValueError):
        return False

    if answer.error or answer.model != model or extract.model != model:
        return False

    from examprep.answer.synthesize import input_hash as answer_hash

    return answer.input_hash == answer_hash(ticket.question, extract, model)


async def _one(
    client: LLMClient,
    slug: str,
    ticket: Ticket,
    chunks: dict[str, Chunk],
    force: bool,
    top_k: int | None,
) -> Answer | None:
    if not force and _is_current(slug, ticket, client.model):
        log.info("answer.skip", question_id=ticket.question.id)
        return None

    extract = await extract_question(client, slug, ticket.question, top_k=top_k)
    return await synthesize_question(
        client,
        slug,
        ticket.question,
        ticket.kind,
        ticket.question_set,
        extract,
        chunks,
    )


async def answer_course(
    slug: str,
    set_id: str | None = None,
    question_id: str | None = None,
    force: bool = False,
    top_k: int | None = None,
    dry_run: bool = False,
) -> list[Answer]:
    tickets = tickets_of(slug, set_id=set_id, question_id=question_id)
    if not tickets:
        raise ValueError("не найдено ни одного вопроса по заданным условиям")

    if dry_run:
        for ticket in tickets:
            status = "готов" if _is_current(slug, ticket, LLMClient().model) else "нужно считать"
            log.info("answer.plan", question_id=ticket.question.id, status=status)
        return []

    if not (course_dir(slug) / "chunks.jsonl").exists():
        raise ValueError(f"нет chunks.jsonl для курса {slug}; сначала `examprep index`")

    chunks = {chunk.chunk_id: chunk for chunk in store.iter_chunks(slug)}
    client = LLMClient()

    answers: list[Answer] = []
    for ticket in tickets:
        try:
            answer = await _one(client, slug, ticket, chunks, force, top_k)
        except Exception:
            # One broken ticket must not cost the other forty-six.
            log.exception("answer.failed", question_id=ticket.question.id)
            continue
        if answer is not None:
            answers.append(answer)

    log.info("answer.done", slug=slug, generated=len(answers), tickets=len(tickets))
    return answers


def run(
    slug: str,
    set_id: str | None = None,
    question_id: str | None = None,
    force: bool = False,
    top_k: int | None = None,
    dry_run: bool = False,
) -> list[Answer]:
    return asyncio.run(
        answer_course(
            slug,
            set_id=set_id,
            question_id=question_id,
            force=force,
            top_k=top_k,
            dry_run=dry_run,
        )
    )
