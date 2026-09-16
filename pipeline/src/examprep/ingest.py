"""Step 1: turn ``sources.txt`` into the ``sources`` list of ``course.json``."""

from __future__ import annotations

from typing import Any

import structlog

from examprep.config import course_dir
from examprep.schemas import Source
from examprep.store import load_course, read_lines, save_course

log = structlog.get_logger()


def _extract(url: str) -> dict[str, Any]:
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    with YoutubeDL(options) as ydl:
        info: dict[str, Any] = ydl.extract_info(url, download=False)
    return info


def _entries(info: dict[str, Any]) -> list[dict[str, Any]]:
    """A playlist yields its entries; a single video yields itself."""

    if info.get("_type") == "playlist":
        return [e for e in info.get("entries") or [] if e]
    return [info]


def ingest(slug: str, force: bool = False) -> list[Source]:
    course = load_course(slug)
    if course.sources and not force:
        log.info("ingest.skip", slug=slug, sources=len(course.sources))
        return course.sources

    urls = read_lines(course_dir(slug) / "sources.txt")
    if not urls:
        raise ValueError(f"sources.txt is empty for course {slug}")

    sources: list[Source] = []
    seen: set[str] = set()
    for url in urls:
        for entry in _entries(_extract(url)):
            video_id = entry["id"]
            if video_id in seen:
                continue
            seen.add(video_id)
            duration = entry.get("duration")
            sources.append(
                Source(
                    video_id=video_id,
                    url=f"https://youtu.be/{video_id}",
                    title=entry.get("title") or video_id,
                    duration_s=float(duration) if duration else None,
                    order=len(sources) + 1,
                )
            )

    course.sources = sources
    save_course(course)
    log.info("ingest.done", slug=slug, sources=len(sources))
    return sources
