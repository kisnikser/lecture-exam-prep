"""Checking that a quote really comes from the fragment it is attributed to.

The whole promise of the project is that every claim can be traced to a moment
in a lecture, so a quote the model invented has to be caught here rather than
shown to the student as something the lecturer said.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

MIN_QUOTE_LEN = 12
FUZZY_THRESHOLD = 0.85
# Shorter coincidences are common between any two Russian sentences and would
# let an invented quote accumulate a passing score out of noise.
MIN_BLOCK_LEN = 4

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Case, punctuation and spacing are not what makes a quote genuine."""

    lowered = text.lower().replace("ё", "е")
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", lowered)).strip()


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
