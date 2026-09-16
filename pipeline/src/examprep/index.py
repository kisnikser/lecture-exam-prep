"""Step 4: cut transcripts into chunks and embed them.

Chunks are windows of wall-clock time rather than of tokens: a citation has to
point at a moment in the video, and a window aligned to segment boundaries keeps
a sentence whole.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from examprep.config import course_dir, get_settings
from examprep.schemas import Chunk, Transcript
from examprep.store import iter_chunks, load_course, load_transcript, save_chunks

log = structlog.get_logger()

EMBEDDINGS_NAME = "embeddings.npy"


def embeddings_path(slug: str) -> Path:
    return course_dir(slug) / EMBEDDINGS_NAME


def chunk_transcript(
    transcript: Transcript,
    window_s: float,
    overlap_s: float,
) -> list[Chunk]:
    segments = transcript.segments
    chunks: list[Chunk] = []
    position = 0

    while position < len(segments):
        start = segments[position].start
        last = position
        while last < len(segments) and segments[last].end - start <= window_s:
            last += 1
        last = max(last, position + 1)  # a segment longer than the window stands alone

        text = " ".join(segments[i].text for i in range(position, last)).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{transcript.video_id}:{len(chunks):04d}",
                    video_id=transcript.video_id,
                    start=start,
                    end=segments[last - 1].end,
                    text=text,
                )
            )

        if last >= len(segments):
            break

        # Step back so the next window overlaps the tail of this one.
        end = segments[last - 1].end
        step_back = last - 1
        while step_back > position and segments[step_back].start > end - overlap_s:
            step_back -= 1
        position = max(step_back, position + 1)

    return chunks


def build_chunks(slug: str) -> list[Chunk]:
    settings = get_settings()
    course = load_course(slug)
    transcripts_dir = course_dir(slug) / "transcripts"

    chunks: list[Chunk] = []
    for source in course.sources:
        if not (transcripts_dir / f"{source.video_id}.json").exists():
            log.warning("index.no_transcript", video_id=source.video_id)
            continue
        transcript = load_transcript(slug, source.video_id)
        video_chunks = chunk_transcript(
            transcript,
            window_s=settings.chunk_seconds,
            overlap_s=settings.chunk_overlap_seconds,
        )
        log.info("index.chunked", video_id=source.video_id, chunks=len(video_chunks))
        chunks.extend(video_chunks)

    return chunks


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    """Dense vectors for the chunk texts, normalized for cosine similarity."""

    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise RuntimeError("sentence-transformers не установлен: `uv sync --extra gpu`") from exc

    device = _device()
    model = SentenceTransformer(model_name or settings.embed_model, device=device)
    log.info("index.embedding", texts=len(texts), device=device)
    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_index(slug: str, force: bool = False, with_embeddings: bool = True) -> int:
    chunks_path = course_dir(slug) / "chunks.jsonl"
    vectors_path = embeddings_path(slug)

    if chunks_path.exists() and not force:
        chunks = list(iter_chunks(slug))
        log.info("index.chunks_reused", chunks=len(chunks))
    else:
        chunks = build_chunks(slug)
        if not chunks:
            raise ValueError(f"нет транскриптов для курса {slug}")
        save_chunks(slug, chunks)

    if with_embeddings and (force or not vectors_path.exists()):
        vectors = embed_texts([c.text for c in chunks])
        np.save(vectors_path, vectors)
        log.info("index.embeddings_saved", shape=list(vectors.shape))

    return len(chunks)
