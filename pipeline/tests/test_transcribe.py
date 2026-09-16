from pathlib import Path

from examprep.transcribe import multi_gpu
from examprep.transcribe.whisper import (
    GLOSSARY_CHAR_LIMIT,
    _segments_from,
    build_initial_prompt,
    model_id,
    resolve_device,
)

SLUG = "hps-skvorchevsky"


def test_cpu_stays_in_float32() -> None:
    assert resolve_device("cpu") == ("cpu", "float32")
    assert resolve_device("cuda") == ("cuda", "float16")


def test_model_id_spells_out_the_repository() -> None:
    assert model_id("large-v3") == "openai/whisper-large-v3"
    assert model_id("openai/whisper-large-v3") == "openai/whisper-large-v3"


def test_open_ended_last_chunk_is_closed_with_the_duration() -> None:
    chunks = [
        {"timestamp": (0.0, 5.0), "text": "первая"},
        {"timestamp": (5.0, None), "text": "последняя"},
    ]
    segments = _segments_from(chunks, duration=12.5)

    assert [s.end for s in segments] == [5.0, 12.5]


def test_chunks_without_a_start_are_dropped() -> None:
    assert _segments_from([{"timestamp": (None, None), "text": "х"}], duration=1.0) == []


def test_initial_prompt_is_built_from_the_glossary() -> None:
    prompt = build_initial_prompt(SLUG)
    assert prompt is not None
    assert prompt.startswith("История и философия науки")
    assert "Скворчевский" in prompt
    assert len(prompt) < GLOSSARY_CHAR_LIMIT + 200


def test_pending_videos_skips_sources_without_audio(monkeypatch) -> None:
    monkeypatch.setattr(multi_gpu, "transcribe_source", lambda slug, video_id: Path("/nonexistent"))
    assert multi_gpu.pending_videos(SLUG) == []
