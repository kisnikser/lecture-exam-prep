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


def test_inverted_chunk_timestamps_are_swapped() -> None:
    # Whisper long-form chunking sometimes returns end < start.
    chunks = [{"timestamp": (2583.44, 2575.0), "text": "всё разрушает."}]
    segments = _segments_from(chunks, duration=5000.0)

    assert len(segments) == 1
    assert segments[0].start == 2575.0
    assert segments[0].end == 2583.44


def test_open_ended_middle_chunk_closes_at_the_next_start() -> None:
    chunks = [
        {"timestamp": (10.0, None), "text": "середина"},
        {"timestamp": (40.0, 50.0), "text": "дальше"},
    ]
    segments = _segments_from(chunks, duration=90.0)

    assert [s.end for s in segments] == [40.0, 50.0]


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


def word(text: str, start: float | None, end: float | None) -> dict:
    return {"text": text, "timestamp": (start, end)}


def test_words_are_cut_on_a_long_pause() -> None:
    from examprep.transcribe.whisper import group_words

    words = [
        word(" Поппер", 0.0, 0.5),
        word(" считает", 0.5, 1.0),
        word(" Кун", 5.0, 5.4),  # пауза 4 с
    ]
    segments = group_words(words, duration=6.0)

    assert [s.text for s in segments] == ["Поппер считает", "Кун"]
    assert segments[0].end == 1.0
    assert segments[1].start == 5.0


def test_a_long_run_without_pauses_is_split_by_length() -> None:
    from examprep.transcribe.whisper import group_words

    words = [word(f" с{i}", i * 0.5, i * 0.5 + 0.5) for i in range(100)]
    segments = group_words(words, duration=50.0, max_segment_s=20.0)

    assert len(segments) > 1
    assert all(s.end - s.start <= 21.0 for s in segments)


def test_words_without_timestamps_join_the_current_segment() -> None:
    from examprep.transcribe.whisper import group_words

    words = [word(" раз", 0.0, 0.5), word(" два", None, None), word(" три", 1.0, 1.5)]
    segments = group_words(words, duration=2.0)

    assert [s.text for s in segments] == ["раз два три"]


def test_inverted_word_bounds_are_swapped() -> None:
    from examprep.transcribe.whisper import group_words

    segments = group_words([word(" слово", 9.0, 8.0)], duration=10.0)

    assert segments[0].start == 8.0
    assert segments[0].end == 9.0


def test_batch_size_per_timestamp_mode() -> None:
    from examprep.transcribe.whisper import BATCH_SIZE, WORD_BATCH_SIZE, _batch_size

    assert _batch_size("chunk") == BATCH_SIZE
    assert _batch_size("word") == WORD_BATCH_SIZE
    assert _batch_size("sequential") == 1


def test_per_gpu_repeats_each_card() -> None:
    from examprep.transcribe.multi_gpu import gpu_slots

    assert gpu_slots([0, 1], per_gpu=3) == [0, 0, 0, 1, 1, 1]
    assert gpu_slots([0, 1, 2]) == [0, 1, 2]
    assert gpu_slots([0], per_gpu=0) == [0]
    assert gpu_slots(None) == []


def test_bounds_past_the_end_of_the_audio_stay_ordered() -> None:
    """The crash from the full run: both timestamps sat beyond the recording."""

    from examprep.transcribe.whisper import _chunk_bounds

    start, end = _chunk_bounds(5334.02, 5318.4145, None, duration=5057.0, last=False)

    assert start <= end
    assert end <= 5057.0


def test_inverted_bounds_inside_the_audio_are_swapped() -> None:
    from examprep.transcribe.whisper import _chunk_bounds

    assert _chunk_bounds(2583.44, 2575.0, None, duration=5000.0, last=False) == (2575.0, 2583.44)


def test_open_end_is_closed_by_the_next_chunk() -> None:
    from examprep.transcribe.whisper import _chunk_bounds

    assert _chunk_bounds(10.0, None, 25.0, duration=5000.0, last=False) == (10.0, 25.0)


def test_open_end_of_the_last_chunk_is_the_duration() -> None:
    from examprep.transcribe.whisper import _chunk_bounds

    assert _chunk_bounds(10.0, None, None, duration=50.0, last=True) == (10.0, 50.0)


def test_zero_length_segment_gets_a_span_from_its_text() -> None:
    """Whisper sometimes reports a paragraph as lasting no time at all."""

    from examprep.transcribe.whisper import SPEECH_CHARS_PER_SECOND, _close_empty_span

    text = "а" * 125
    end = _close_empty_span(100.0, 100.0, text, next_start=None, duration=5000.0)

    assert end == 100.0 + 125 / SPEECH_CHARS_PER_SECOND


def test_estimated_span_does_not_reach_the_next_segment() -> None:
    from examprep.transcribe.whisper import _close_empty_span

    end = _close_empty_span(100.0, 100.0, "а" * 1000, next_start=103.0, duration=5000.0)

    assert end == 103.0


def test_a_real_span_is_left_alone() -> None:
    from examprep.transcribe.whisper import _close_empty_span

    assert _close_empty_span(10.0, 25.0, "речь", next_start=30.0, duration=100.0) == 25.0


def test_an_empty_segment_stays_empty() -> None:
    from examprep.transcribe.whisper import _close_empty_span

    assert _close_empty_span(10.0, 10.0, "   ", next_start=None, duration=100.0) == 10.0
