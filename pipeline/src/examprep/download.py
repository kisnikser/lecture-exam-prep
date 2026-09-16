"""Step 2: download the audio track of every source into ``audio/``.

Audio never reaches git; it travels to the GPU server over rsync (or tar).
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import structlog

from examprep.config import course_dir
from examprep.store import load_course

log = structlog.get_logger()

AUDIO_EXT = "m4a"
WAV_EXT = "wav"
SAMPLE_RATE = 16000


def audio_path(slug: str, video_id: str) -> Path:
    return course_dir(slug) / "audio" / f"{video_id}.{AUDIO_EXT}"


def wav_path(slug: str, video_id: str) -> Path:
    return course_dir(slug) / "audio" / f"{video_id}.{WAV_EXT}"


def transcribe_source(slug: str, video_id: str) -> Path:
    """The prepared 16 kHz WAV when it exists, otherwise the original download.

    Whisper decodes compressed audio through PyAV in a single thread, which on
    a full lecture takes longer than the transcription itself. ffmpeg does the
    same job in seconds, so `prepare-audio` converts once and Whisper then reads
    plain PCM.
    """

    wav = wav_path(slug, video_id)
    return wav if wav.exists() else audio_path(slug, video_id)


def prepare_audio(slug: str, force: bool = False, jobs: int = 8) -> list[Path]:
    """Convert every downloaded track to 16 kHz mono WAV with ffmpeg."""

    course = load_course(slug)

    def convert(video_id: str) -> Path | None:
        source = audio_path(slug, video_id)
        target = wav_path(slug, video_id)
        if not source.exists():
            log.warning("prepare_audio.no_source", video_id=video_id)
            return None
        if target.exists() and target.stat().st_size > 0 and not force:
            return target

        # Absolute paths keep ffmpeg from reading a leading "-" in a video id
        # as an option.
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source.resolve()),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(target.resolve()),
            ],
            check=True,
        )
        log.info("prepare_audio.done", video_id=video_id)
        return target

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(convert, [s.video_id for s in course.sources]))

    return [path for path in results if path is not None]


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
