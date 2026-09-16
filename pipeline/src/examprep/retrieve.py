"""Hybrid retrieval over the chunks of one course.

Dense search alone misses the lecturer's exact wording; BM25 alone misses
paraphrase. The two rankings are merged with reciprocal rank fusion, which needs
no score calibration between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import structlog

from examprep.config import get_settings
from examprep.index import embeddings_path
from examprep.schemas import Chunk
from examprep.store import iter_chunks

log = structlog.get_logger()

RRF_K = 60
WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass
class Candidate:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@lru_cache(maxsize=1)
def _morph() -> object | None:
    try:
        import pymorphy3
    except ImportError:  # pragma: no cover - dependency is declared
        return None
    analyzer: object = pymorphy3.MorphAnalyzer()
    return analyzer


def lemmatize(text: str) -> list[str]:
    """Words reduced to their normal form, so «Поппера» matches «Поппер»."""

    words = WORD_RE.findall(text.lower())
    morph = _morph()
    if morph is None:
        return words
    return [morph.parse(word)[0].normal_form for word in words]  # type: ignore[attr-defined]


def _rrf(
    ranking: list[int], scores: dict[int, float], field: str, ranks: dict[int, dict[str, int]]
) -> None:
    for position, index in enumerate(ranking):
        scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + position + 1)
        ranks.setdefault(index, {})[field] = position + 1


def _bm25_ranking(chunks: list[Chunk], query: str, top_k: int) -> list[int]:
    from rank_bm25 import BM25Okapi

    corpus = [lemmatize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(lemmatize(query))
    return [int(i) for i in np.argsort(scores)[::-1][:top_k] if scores[i] > 0]


def _dense_ranking(slug: str, query: str, top_k: int) -> list[int]:
    path = embeddings_path(slug)
    if not path.exists():
        log.warning("retrieve.no_embeddings", path=str(path))
        return []

    from examprep.index import embed_texts

    vectors = np.load(path)
    query_vector = embed_texts([query])[0]
    similarity = vectors @ query_vector
    return [int(i) for i in np.argsort(similarity)[::-1][:top_k]]


def search(slug: str, query: str, top_k: int | None = None) -> list[Candidate]:
    top_k = top_k or get_settings().retrieve_top_k
    chunks = list(iter_chunks(slug))
    if not chunks:
        raise ValueError(f"нет chunks.jsonl для курса {slug}; сначала `examprep index`")

    pool = top_k * 2
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    _rrf(_dense_ranking(slug, query, pool), scores, "dense", ranks)
    _rrf(_bm25_ranking(chunks, query, pool), scores, "bm25", ranks)

    best = sorted(scores, key=lambda index: scores[index], reverse=True)[:top_k]
    return [
        Candidate(
            chunk=chunks[index],
            score=scores[index],
            dense_rank=ranks[index].get("dense"),
            bm25_rank=ranks[index].get("bm25"),
        )
        for index in best
    ]


def with_context(slug: str, candidate: Candidate, neighbours: int = 1) -> str:
    """The candidate's text padded with its neighbours in the same lecture.

    The quote an answer cites must come from the chunk itself; the neighbours are
    there only so the model can tell what the lecturer was talking about.
    """

    chunks = list(iter_chunks(slug))
    position = next(
        (i for i, chunk in enumerate(chunks) if chunk.chunk_id == candidate.chunk.chunk_id),
        None,
    )
    if position is None:
        return candidate.chunk.text

    window = [
        chunk
        for chunk in chunks[max(0, position - neighbours) : position + neighbours + 1]
        if chunk.video_id == candidate.chunk.video_id
    ]
    return " ".join(chunk.text for chunk in window)
