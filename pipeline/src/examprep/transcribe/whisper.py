"""Step 3: transcribe one prepared lecture with Whisper on the torch stack.

CTranslate2 — the engine behind faster-whisper — ships CUDA 12 builds only,
while this cluster provides CUDA 13. Transcription therefore runs through
transformers on the torch that the shared environment already has.

Input is always the 16 kHz WAV produced by ``prepare-audio``: without
CTranslate2 there is no decoder for compressed audio in this process.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from examprep.clean import clean_segments, load_replacements
from examprep.config import course_dir, get_settings
from examprep.download import SAMPLE_RATE, wav_path
from examprep.schemas import Segment, Transcript
from examprep.store import load_course, read_lines

log = structlog.get_logger()

# Whisper reads at most ~224 tokens of prompt; the rest is ignored.
GLOSSARY_CHAR_LIMIT = 700
HF_MODEL_PREFIX = "openai/whisper-"
CHUNK_LENGTH_S = 30
BATCH_SIZE = 16
BEAM_SIZE = 5

_PIPELINES: dict[tuple[str, str, str], Any] = {}


def model_id(name: str) -> str:
    """`large-v3` is a Hugging Face repository, spelled out."""

    return name if "/" in name else f"{HF_MODEL_PREFIX}{name}"


def cuda_device_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:  # сломанная установка CUDA не должна валить запуск
        return 0


def resolve_device(device: str | None = None) -> tuple[str, str]:
    """Pick the device and the matching dtype (CPU boxes stay in float32)."""

    if device is None:
        device = "cuda" if cuda_device_count() > 0 else "cpu"
    return device, "float16" if device == "cuda" else "float32"


def load_wav(path: Path) -> np.ndarray | None:
    """Read a prepared 16 kHz mono WAV into the array Whisper wants.

    Returns None when the file is not that layout, so the caller can say what
    is wrong instead of feeding the model noise.
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


def _pipeline(name: str, device: str, dtype: str) -> Any:
    key = (name, device, dtype)
    if key not in _PIPELINES:
        import torch
        from transformers import pipeline

        log.info("whisper.load", model=name, device=device, dtype=dtype)
        _PIPELINES[key] = pipeline(
            "automatic-speech-recognition",
            model=name,
            torch_dtype=getattr(torch, dtype),
            device=device,
            chunk_length_s=CHUNK_LENGTH_S,
            batch_size=BATCH_SIZE,
            return_timestamps=True,
        )
    return _PIPELINES[key]


def _segments_from(chunks: list[dict[str, Any]], duration: float) -> list[Segment]:
    """Turn Whisper's chunk timestamps into segments.

    The final chunk of a long recording sometimes comes back with an open end;
    it is closed with the length of the audio.
    """

    segments: list[Segment] = []
    for chunk in chunks:
        start, end = chunk.get("timestamp") or (None, None)
        if start is None:
            continue
        text = str(chunk.get("text", ""))
        segments.append(Segment(start=float(start), end=float(end or duration), text=text))
    return segments


def transcribe_one(
    slug: str,
    video_id: str,
    model_size: str | None = None,
    device: str | None = None,
) -> Transcript:
    settings = get_settings()
    name = model_id(model_size or settings.whisper_model)
    device, dtype = resolve_device(device)

    path = wav_path(slug, video_id)
    audio = load_wav(path) if path.exists() else None
    if audio is None:
        raise FileNotFoundError(
            f"нет подготовленного WAV для {video_id}: {path}. Сначала `examprep prepare-audio`."
        )

    generate_kwargs: dict[str, Any] = {
        "language": "ru",
        "task": "transcribe",
        "num_beams": BEAM_SIZE,
    }
    prompt = build_initial_prompt(slug)
    if prompt:
        generate_kwargs["prompt_ids"] = _prompt_ids(name, prompt, device)

    result = _pipeline(name, device, dtype)(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        generate_kwargs=generate_kwargs,
    )

    duration = len(audio) / SAMPLE_RATE
    segments = _segments_from(result.get("chunks") or [], duration)
    cleaned = clean_segments(segments, load_replacements(slug))
    log.info(
        "whisper.done",
        video_id=video_id,
        segments=len(cleaned),
        dropped=len(segments) - len(cleaned),
        duration_s=round(duration),
    )

    return Transcript(
        video_id=video_id,
        source="whisper",
        model=name,
        language="ru",
        segments=cleaned,
    )


def _prompt_ids(name: str, prompt: str, device: str) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    return processor.get_prompt_ids(prompt, return_tensors="pt").to(device)
