"""Checking that a quote really comes from the fragment it is attributed to.

The whole promise of the project is that every claim can be traced to a moment
in a lecture, so a quote the model invented has to be caught here rather than
shown to the student as something the lecturer said.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

from examprep.schemas import Segment

MIN_QUOTE_LEN = 12
FUZZY_THRESHOLD = 0.85
# Shorter coincidences are common between any two Russian sentences and would
# let an invented quote accumulate a passing score out of noise.
MIN_BLOCK_LEN = 4
# Enough of the quote's opening to place it, short enough to survive the
# model tidying the tail of the sentence.
OPENING_LEN = 40

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Case, punctuation and spacing are not what makes a quote genuine."""

    lowered = text.lower().replace("ё", "е")
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", lowered)).strip()


def quote_start(segments: Sequence[Segment], quote: str, fallback: float) -> float:
    """When inside the chunk the quote begins, in seconds.

    A chunk spans a minute and a half, so linking to its start can leave the
    student waiting through the whole window before hearing the sentence. The
    quote is located among the segments the chunk was built from, and the link
    points at the segment where it starts. Segments are checked in pairs, since
    a quote often runs across a boundary.
    """

    opening = normalize(quote)[:OPENING_LEN]
    if len(opening) < MIN_QUOTE_LEN:
        return fallback

    for index, segment in enumerate(segments):
        own = normalize(segment.text)
        window = own
        if index + 1 < len(segments):
            window = f"{own} {normalize(segments[index + 1].text)}"

        position = window.find(opening)
        # Found in the pair, but starting inside the next segment: that segment
        # gets its own turn, and pointing here would be a segment too early.
        if position >= 0 and position < len(own):
            return segment.start

    return fallback


def quote_matches(quote: str, source: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """Whether ``quote`` is present in ``source``, allowing for small drift.

    The model is asked to copy verbatim but tends to fix the lecturer's grammar
    or drop a filler word, so an exact match is too strict; an unrelated
    sentence still falls far below the threshold.
    """

    needle = normalize(quote)
    haystack = normalize(source)
    if len(needle) < MIN_QUOTE_LEN or not haystack:
        return False
    if needle in haystack:
        return True

    # How much of the quote is found in the fragment, counting every matching
    # run rather than the single longest one: a model that "corrects" a word in
    # the middle otherwise fails on an otherwise verbatim quote.
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(
        block.size for block in matcher.get_matching_blocks() if block.size >= MIN_BLOCK_LEN
    )
    return matched / len(needle) >= threshold
