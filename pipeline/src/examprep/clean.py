"""Cleaning of raw Whisper output.

Whisper invents text on silence and loops on repeated phrases; both would end up
quoted in an answer, so they are dropped before anything is indexed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import structlog

from examprep.config import course_dir
from examprep.schemas import Segment
from examprep.store import read_lines

log = structlog.get_logger()

REPLACEMENTS_SEPARATOR = "->"

# Phrases Whisper adds on silence or music, taken from YouTube subtitle credits.
HALLUCINATIONS = [
    r"субтитры .*",
    r"редактор субтитров .*",
    r"продолжение следует.*",
    r"спасибо за просмотр.*",
    r"подписывайтесь на канал.*",
    r"ставьте лайки.*",
    r"добро пожаловать на наш канал.*",
    r"с вами был.*",
    r"all rights reserved.*",
    r"thanks for watching.*",
    r"[!?.…\-\s]*",
    # Subtitle credits: a role followed by an initial and a surname, as in
    # "Корректор А.Егорова". The initial is what keeps this from matching a
    # lecture that happens to mention an editor.
    r".*\b(?:корректор|редактор|расшифровка|перевод)\s+[а-яё]\.\s*[а-яё-]+.*",
]

HALLUCINATION_RE = re.compile(rf"^(?:{'|'.join(HALLUCINATIONS)})$", re.IGNORECASE)


def load_replacements(slug: str) -> dict[str, str]:
    """Course-specific fixes, one ``неверно -> верно`` per line."""

    replacements: dict[str, str] = {}
    for line in read_lines(course_dir(slug) / "replacements.txt"):
        if REPLACEMENTS_SEPARATOR not in line:
            log.warning("clean.bad_replacement", line=line)
            continue
        wrong, right = line.split(REPLACEMENTS_SEPARATOR, 1)
        replacements[wrong.strip()] = right.strip()
    return replacements


def apply_replacements(text: str, replacements: Mapping[str, str]) -> str:
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


GLOSSARY_ECHO_SHARE = 0.6
GLOSSARY_MIN_TERM_LEN = 5


def is_glossary_echo(
    text: str,
    terms: Sequence[str],
    share: float = GLOSSARY_ECHO_SHARE,
) -> bool:
    """Whether the segment is the glossary prompt read back as speech.

    Whisper is given the course glossary as a prompt and sometimes transcribes
    the prompt itself, which would put names into the lecturer's mouth and from
    there into a citation. A run of glossary terms filling most of a segment is
    that echo; a sentence that merely mentions one philosopher is not.
    """

    lowered = text.lower()
    if not lowered:
        return False

    covered = 0
    matched = 0
    for term in terms:
        normalized = term.lower().strip()
        if len(normalized) < GLOSSARY_MIN_TERM_LEN:
            continue
        occurrences = lowered.count(normalized)
        if occurrences:
            matched += 1
            covered += len(normalized) * occurrences

    return matched >= 2 and covered / len(lowered) >= share


def clean_segments(
    segments: list[Segment],
    replacements: Mapping[str, str] | None = None,
    max_repeats: int = 2,
    glossary: Sequence[str] | None = None,
) -> list[Segment]:
    """Normalize whitespace, drop hallucinations and collapse looped segments."""

    cleaned: list[Segment] = []
    repeats = 0

    for segment in segments:
        text = normalize(segment.text)
        if replacements:
            text = apply_replacements(text, replacements)
        if not text or HALLUCINATION_RE.match(text):
            continue
        if glossary and is_glossary_echo(text, glossary):
            log.info("clean.glossary_echo", text=text[:60])
            continue

        previous = cleaned[-1].text if cleaned else None
        if text == previous:
            repeats += 1
            if repeats >= max_repeats:
                continue
        else:
            repeats = 0

        cleaned.append(Segment(start=segment.start, end=segment.end, text=text))

    return cleaned
