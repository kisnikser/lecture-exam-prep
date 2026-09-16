from pathlib import Path

from examprep.transcribe import multi_gpu
from examprep.transcribe.whisper import GLOSSARY_CHAR_LIMIT, build_initial_prompt, resolve_device

SLUG = "hps-skvorchevsky"


def test_cpu_falls_back_to_int8() -> None:
    assert resolve_device("cpu") == ("cpu", "int8")
    assert resolve_device("cuda") == ("cuda", "float16")


def test_initial_prompt_is_built_from_the_glossary() -> None:
    prompt = build_initial_prompt(SLUG)
    assert prompt is not None
    assert prompt.startswith("История и философия науки")
    assert "Скворчевский" in prompt
    assert len(prompt) < GLOSSARY_CHAR_LIMIT + 200


def test_pending_videos_skips_sources_without_audio(monkeypatch) -> None:
    monkeypatch.setattr(multi_gpu, "transcribe_source", lambda slug, video_id: Path("/nonexistent"))
    assert multi_gpu.pending_videos(SLUG) == []
