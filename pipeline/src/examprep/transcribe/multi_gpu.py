"""Run transcription over a pool of GPUs, one worker process per device.

``CUDA_VISIBLE_DEVICES`` has to be set before CTranslate2 loads, so each worker
claims its device in the pool initializer and then sees it as device 0.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.queues import Queue as MpQueue
from queue import Empty

import structlog

from examprep.config import course_dir
from examprep.download import audio_path
from examprep.store import load_course, save_transcript

log = structlog.get_logger()

_GPU_QUEUE: MpQueue[int] | None = None


def pending_videos(slug: str, force: bool = False, limit: int | None = None) -> list[str]:
    """Videos that have audio but no transcript yet."""

    course = load_course(slug)
    transcripts_dir = course_dir(slug) / "transcripts"

    pending = []
    for source in course.sources:
        if not audio_path(slug, source.video_id).exists():
            log.warning("transcribe.no_audio", video_id=source.video_id)
            continue
        if not force and (transcripts_dir / f"{source.video_id}.json").exists():
            continue
        pending.append(source.video_id)

    return pending[:limit] if limit else pending


def _init_worker(gpu_queue: MpQueue[int]) -> None:
    global _GPU_QUEUE
    _GPU_QUEUE = gpu_queue
    try:
        gpu = gpu_queue.get_nowait()
    except Empty:  # pragma: no cover - only if more workers than GPUs
        gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _work(task: tuple[str, str, str | None]) -> str:
    slug, video_id, model_size = task
    from examprep.transcribe.whisper import transcribe_one

    transcript = transcribe_one(slug, video_id, model_size=model_size)
    save_transcript(slug, transcript)
    return video_id


def transcribe_course(
    slug: str,
    gpus: list[int] | None = None,
    model_size: str | None = None,
    force: bool = False,
    limit: int | None = None,
) -> list[str]:
    videos = pending_videos(slug, force=force, limit=limit)
    if not videos:
        log.info("transcribe.nothing_to_do", slug=slug)
        return []

    tasks = [(slug, video_id, model_size) for video_id in videos]
    workers = min(len(gpus or []), len(tasks))

    if workers <= 1:
        if gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        return [_work(task) for task in tasks]

    context = mp.get_context("spawn")
    gpu_queue: MpQueue[int] = context.Queue()
    for gpu in (gpus or [])[:workers]:
        gpu_queue.put(gpu)

    with context.Pool(workers, initializer=_init_worker, initargs=(gpu_queue,)) as pool:
        done = list(pool.imap_unordered(_work, tasks))

    log.info("transcribe.done", slug=slug, videos=len(done), workers=workers)
    return done
