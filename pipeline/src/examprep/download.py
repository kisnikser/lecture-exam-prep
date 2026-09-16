"""Step 2: download the audio track of every source into ``audio/``.

Audio never reaches git; it travels to the GPU server over rsync.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from examprep.config import course_dir
from examprep.store import load_course

log = structlog.get_logger()

AUDIO_EXT = "m4a"


def audio_path(slug: str, video_id: str) -> Path:
    return course_dir(slug) / "audio" / f"{video_id}.{AUDIO_EXT}"


def download(slug: str, force: bool = False, limit: int | None = None) -> list[Path]:
    from yt_dlp import YoutubeDL

    course = load_course(slug)
    if not course.sources:
        raise ValueError(f"course {slug} has no sources; run `examprep ingest` first")

    out_dir = course_dir(slug) / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = course.sources[:limit] if limit else course.sources
    downloaded: list[Path] = []
    for source in sources:
        target = audio_path(slug, source.video_id)
        if target.exists() and not force:
            log.info("download.skip", video_id=source.video_id)
            downloaded.append(target)
            continue

        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "format": f"bestaudio[ext={AUDIO_EXT}]/bestaudio",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "retries": 5,
        }
        with YoutubeDL(options) as ydl:
            ydl.download([source.url])

        found = next(out_dir.glob(f"{source.video_id}.*"), None)
        if found is None:
            raise RuntimeError(f"no audio file produced for {source.video_id}")
        if found != target:
            found = found.rename(target)
        log.info("download.done", video_id=source.video_id, path=str(found))
        downloaded.append(found)

    return downloaded
