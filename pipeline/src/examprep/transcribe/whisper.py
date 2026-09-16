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

# Word timestamps come from cross-attention alignment rather than from what the
# model predicts, so segments can be cut where the lecturer actually pauses.
# Alignment keeps the attention weights of the whole batch, which at
# BATCH_SIZE overflows an 80 GB card, hence the smaller batch here.
WORD_PAUSE_S = 0.6
MAX_SEGMENT_S = 20.0
WORD_BATCH_SIZE = 4

_PIPELINES: dict[tuple[str, str, str, str], Any] = {}


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


def _pipeline(name: str, device: str, dtype: str, timestamps: str) -> Any:
    """Whisper as a transformers pipeline, in one of three timestamp modes.

    ``chunk`` cuts fixed 30-second windows and transcribes them independently:
    fast, because windows batch, but each window predicts timestamps on its own
    scale and they do not stitch back into a coherent one.

    ``sequential`` is Whisper's own long-form algorithm — the next window starts
    where the model says the last utterance ended, carrying the text along as
    context. Timestamps stay coherent across the file; nothing batches inside it.

    ``word`` keeps the fast windows but aligns every word against the decoder's
    cross-attention, which is the most precise and the most expensive.
    """

    key = (name, device, dtype, timestamps)
    if key not in _PIPELINES:
        import torch
        from transformers import pipeline

        options: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype),
            "device": device,
            "return_timestamps": "word" if timestamps == "word" else True,
        }
        if timestamps != "sequential":
            options["chunk_length_s"] = CHUNK_LENGTH_S

        log.info("whisper.load", model=name, device=device, dtype=dtype, timestamps=timestamps)
        _PIPELINES[key] = pipeline("automatic-speech-recognition", model=name, **options)
    return _PIPELINES[key]


def _batch_size(timestamps: str) -> int:
    """Sequential decoding has nothing to batch: each window follows the last."""

    if timestamps == "sequential":
        return 1
    return WORD_BATCH_SIZE if timestamps == "word" else BATCH_SIZE


def _run(pipe: Any, audio: np.ndarray, generate_kwargs: dict[str, Any], batch: int) -> Any:
    """Transcribe, halving the batch whenever the card runs out of memory."""

    import torch

    while True:
        try:
            return pipe(
                {"array": audio, "sampling_rate": SAMPLE_RATE},
                generate_kwargs=generate_kwargs,
                batch_size=batch,
            )
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            log.warning("whisper.oom_retry", batch_size=batch)


def _span(start: float, end: float | None, duration: float) -> tuple[float, float]:
    """A single word's bounds, clamped and un-inverted."""

    begin = max(0.0, float(start))
    finish = begin if end is None else float(end)
    if finish < begin:
        begin, finish = finish, begin
    return min(begin, duration), min(finish, duration)


def group_words(
    words: list[dict[str, Any]],
    duration: float,
    pause_s: float = WORD_PAUSE_S,
    max_segment_s: float = MAX_SEGMENT_S,
) -> list[Segment]:
    """Collect word-level timestamps into segments, cutting on pauses.

    A new segment starts when the lecturer pauses longer than ``pause_s`` or
    when the current one has run for ``max_segment_s``, which keeps segments
    short enough for a citation to point at the right moment.
    """

    segments: list[Segment] = []
    buffer: list[str] = []
    seg_start: float | None = None
    previous_end: float | None = None

    def flush() -> None:
        nonlocal buffer, seg_start, previous_end
        text = "".join(buffer).strip()
        if text and seg_start is not None:
            end = max(previous_end if previous_end is not None else seg_start, seg_start)
            segments.append(Segment(start=seg_start, end=min(end, duration), text=text))
        buffer = []
        seg_start = None

    for word in words:
        start, end = word.get("timestamp") or (None, None)
        text = str(word.get("text", ""))
        if start is None:
            buffer.append(text)
            continue

        begin, finish = _span(start, end, duration)
        if seg_start is None:
            seg_start = begin
        elif begin - (previous_end or begin) > pause_s or finish - seg_start > max_segment_s:
            flush()
            seg_start = begin

        buffer.append(text)
        previous_end = finish

    flush()
    return segments


def _chunk_bounds(
    start: float,
    end: float | None,
    next_start: float | None,
    duration: float,
    last: bool,
) -> tuple[float, float]:
    """Whisper's long-form chunks often miss or invert the end timestamp."""

    start = max(0.0, start)
    if end is not None:
        closed = float(end)
        if closed < start:
            start, closed = closed, start
        return start, min(closed, duration)
    if last:
        return start, duration
    if next_start is not None and next_start >= start:
        return start, min(float(next_start), duration)
    return start, start


def _segments_from(chunks: list[dict[str, Any]], duration: float) -> list[Segment]:
    """Turn Whisper's chunk timestamps into segments.

    The final chunk of a long recording sometimes comes back with an open end;
    it is closed with the length of the audio. A middle chunk may invert
    start/end or omit the end entirely — both happen with ``chunk_length_s``.
    """

    segments: list[Segment] = []
    for index, chunk in enumerate(chunks):
        start, end = chunk.get("timestamp") or (None, None)
        if start is None:
            continue
        next_start = None
        if index + 1 < len(chunks):
            nxt = (chunks[index + 1].get("timestamp") or (None, None))[0]
            next_start = float(nxt) if nxt is not None else None
        start_s, end_s = _chunk_bounds(
            float(start),
            float(end) if end is not None else None,
            next_start,
            duration,
            last=index == len(chunks) - 1,
        )
        segments.append(Segment(start=start_s, end=end_s, text=str(chunk.get("text", ""))))
    return segments


def transcribe_one(
    slug: str,
    video_id: str,
    model_size: str | None = None,
    device: str | None = None,
    timestamps: str = "word",
    glossary_prompt: bool = False,
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
    # The glossary is off by default. Windowed decoding applies the prompt to
    # every 30-second window, and the model transcribes the prompt itself often
    # enough to matter: measured on one lecture it swallowed 19% of the timeline
    # and replaced the speech there with lists of names. Misheard names are
    # fixed afterwards through replacements.txt, which cannot eat the audio.
    if glossary_prompt:
        prompt = build_initial_prompt(slug)
        if prompt:
            generate_kwargs["prompt_ids"] = _prompt_ids(name, prompt, device)

    result = _run(
        _pipeline(name, device, dtype, timestamps),
        audio,
        generate_kwargs,
        _batch_size(timestamps),
    )

    duration = len(audio) / SAMPLE_RATE
    chunks = result.get("chunks") or []
    segments = (
        group_words(chunks, duration) if timestamps == "word" else _segments_from(chunks, duration)
    )
    cleaned = clean_segments(
        segments,
        load_replacements(slug),
        glossary=read_lines(course_dir(slug) / "glossary.txt"),
    )
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
