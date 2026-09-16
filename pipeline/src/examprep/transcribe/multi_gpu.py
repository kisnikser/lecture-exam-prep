"""Run transcription over a pool of GPUs, one worker process per device.

``CUDA_VISIBLE_DEVICES`` has to be set before torch loads, so each worker
claims its device in the pool initializer and then sees it as device 0.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.queues import Queue as MpQueue
from queue import Empty

import structlog

from examprep.config import course_dir
from examprep.download import transcribe_source
from examprep.store import load_course, save_transcript

log = structlog.get_logger()

_GPU_QUEUE: MpQueue[int] | None = None


def pending_videos(slug: str, force: bool = False, limit: int | None = None) -> list[str]:
    """Videos that have audio but no transcript yet."""

    course = load_course(slug)
    transcripts_dir = course_dir(slug) / "transcripts"

    pending = []
    for source in course.sources:
        if not transcribe_source(slug, source.video_id).exists():
            log.warning("transcribe.no_audio", video_id=source.video_id)
            continue
        if not force and (transcripts_dir / f"{source.video_id}.json").exists():
            continue
        pending.append(source.video_id)

    return pending[:limit] if limit else pending


def gpu_slots(gpus: list[int] | None, per_gpu: int = 1) -> list[int]:
    """One entry per worker process, naming the card it will claim.

    A sequential lecture leaves the card less than half busy, so several can
    share one GPU; each worker still loads its own copy of the model, which is
    why the count is capped by memory rather than raised without measuring.
    """

    return [gpu for gpu in (gpus or []) for _ in range(max(1, per_gpu))]


def _init_worker(gpu_queue: MpQueue[int]) -> None:
    global _GPU_QUEUE
    _GPU_QUEUE = gpu_queue
    try:
        gpu = gpu_queue.get_nowait()
    except Empty:  # pragma: no cover - only if more workers than GPUs
        gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Word alignment allocates in bursts; expandable segments keep the
    # allocator from fragmenting itself into an out-of-memory error.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _work(task: tuple[str, str, str | None, str, bool]) -> str | None:
    slug, video_id, model_size, timestamps, glossary_prompt = task
    from examprep.transcribe.whisper import transcribe_one

    try:
        transcript = transcribe_one(
            slug,
            video_id,
            model_size=model_size,
            timestamps=timestamps,
            glossary_prompt=glossary_prompt,
        )
        save_transcript(slug, transcript)
    except Exception:
        log.exception("transcribe.failed", video_id=video_id)
        return None
    return video_id


def transcribe_course(
    slug: str,
    gpus: list[int] | None = None,
    model_size: str | None = None,
    force: bool = False,
    limit: int | None = None,
    video_id: str | None = None,
    timestamps: str = "word",
    per_gpu: int = 1,
    glossary_prompt: bool = False,
) -> list[str]:
    videos = pending_videos(slug, force=force, limit=limit)
    if video_id is not None:
        videos = [v for v in videos if v == video_id]
        if not videos:
            log.warning("transcribe.video_not_pending", video_id=video_id)
    if not videos:
        log.info("transcribe.nothing_to_do", slug=slug)
        return []

    tasks = [(slug, video_id, model_size, timestamps, glossary_prompt) for video_id in videos]
    slots = gpu_slots(gpus, per_gpu)
    workers = min(len(slots), len(tasks))

    if workers <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        done = [video_id for task in tasks if (video_id := _work(task))]
    else:
        context = mp.get_context("spawn")
        gpu_queue: MpQueue[int] = context.Queue()
        for gpu in slots[:workers]:
            gpu_queue.put(gpu)

        with context.Pool(workers, initializer=_init_worker, initargs=(gpu_queue,)) as pool:
            done = [video_id for video_id in pool.imap_unordered(_work, tasks) if video_id]

    log.info("transcribe.done", slug=slug, videos=len(done), workers=workers)
    return done
