"""Step 3: transcribe one audio file with faster-whisper."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import structlog

from examprep.clean import clean_segments, load_replacements
from examprep.config import course_dir, get_settings
from examprep.download import SAMPLE_RATE, transcribe_source
from examprep.schemas import Segment, Transcript
from examprep.store import load_course, read_lines

log = structlog.get_logger()

# Whisper reads at most ~224 tokens of initial_prompt; the rest is ignored.
GLOSSARY_CHAR_LIMIT = 700
WAV_SUFFIX = "wav"

_MODELS: dict[tuple[str, str, str], object] = {}


def cuda_device_count() -> int:
    """GPUs visible to CTranslate2, without importing torch."""

    try:
        import ctranslate2
    except ImportError:
        return 0
    try:
        return int(ctranslate2.get_cuda_device_count())
    except Exception:  # сломанная установка CUDA не должна валить запуск
        return 0


def resolve_device(device: str | None = None) -> tuple[str, str]:
    """Pick the device and the matching compute type (CPU boxes get int8)."""

    if device is None:
        device = "cuda" if cuda_device_count() > 0 else "cpu"
    return device, "float16" if device == "cuda" else "int8"


def build_initial_prompt(slug: str) -> str | None:
    """Course title plus as much of the glossary as Whisper will read."""

    terms = read_lines(course_dir(slug) / "glossary.txt")
    if not terms:
        return None

    listed = ""
    for term in terms:
        candidate = f"{listed}, {term}" if listed else term
        if len(candidate) > GLOSSARY_CHAR_LIMIT:
            break
        listed = candidate

    return f"{load_course(slug).title}. Имена и термины: {listed}."


def load_wav(path: Path) -> np.ndarray | None:
    """Read a prepared 16 kHz mono WAV straight into the array Whisper wants.

    faster-whisper decodes through PyAV, which on a full lecture costs minutes
    of single-threaded work while the GPU waits. Our own WAV has a known layout,
    so reading it is a buffer copy. Returns None if the file is not that layout,
    and the caller falls back to letting faster-whisper open it.
    """

    try:
        with wave.open(str(path), "rb") as handle:
            if (
                handle.getnchannels() != 1
                or handle.getframerate() != SAMPLE_RATE
                or handle.getsampwidth() != 2
            ):
                return None
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        log.warning("whisper.wav_unreadable", path=str(path), error=str(exc))
        return None

    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _model(model_size: str, device: str, compute_type: str) -> object:
    key = (model_size, device, compute_type)
    if key not in _MODELS:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise RuntimeError(
                "faster-whisper не установлен: `uv sync --extra asr` (или `--extra gpu`)"
            ) from exc
        log.info("whisper.load", model=model_size, device=device, compute_type=compute_type)
        _MODELS[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODELS[key]


def transcribe_one(
    slug: str,
    video_id: str,
    model_size: str | None = None,
    device: str | None = None,
) -> Transcript:
    settings = get_settings()
    model_size = model_size or settings.whisper_model
    device, compute_type = resolve_device(device)

    path = transcribe_source(slug, video_id)
    if not path.exists():
        raise FileNotFoundError(f"нет аудио для {video_id}: {path}")

    model = _model(model_size, device, compute_type)
    audio = load_wav(path) if path.suffix == f".{WAV_SUFFIX}" else None
    raw, info = model.transcribe(  # type: ignore[attr-defined]
        audio if audio is not None else str(path),
        language="ru",
        beam_size=5,
        vad_filter=True,
        initial_prompt=build_initial_prompt(slug),
    )

    segments = [
        Segment(start=float(s.start), end=float(s.end), text=s.text) for s in raw if s.text.strip()
    ]
    cleaned = clean_segments(segments, load_replacements(slug))
    log.info(
        "whisper.done",
        video_id=video_id,
        segments=len(cleaned),
        dropped=len(segments) - len(cleaned),
        duration_s=round(getattr(info, "duration", 0.0)),
    )

    return Transcript(
        video_id=video_id,
        source="whisper",
        model=model_size,
        language="ru",
        segments=cleaned,
    )
