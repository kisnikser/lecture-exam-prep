from examprep.index import chunk_transcript
from examprep.schemas import Segment, Transcript


def transcript(segments: list[tuple[float, float, str]]) -> Transcript:
    return Transcript(
        video_id="abc123",
        source="whisper",
        language="ru",
        segments=[Segment(start=s, end=e, text=t) for s, e, t in segments],
    )


def test_chunks_cover_every_segment_and_are_numbered() -> None:
    source = transcript([(i * 10.0, i * 10.0 + 10.0, f"фраза {i}") for i in range(20)])
    chunks = chunk_transcript(source, window_s=90.0, overlap_s=15.0)

    assert [c.chunk_id for c in chunks] == [f"abc123:{i:04d}" for i in range(len(chunks))]
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 200.0
    joined = " ".join(c.text for c in chunks)
    for i in range(20):
        assert f"фраза {i}" in joined


def test_windows_overlap() -> None:
    source = transcript([(i * 10.0, i * 10.0 + 10.0, f"фраза {i}") for i in range(20)])
    chunks = chunk_transcript(source, window_s=90.0, overlap_s=15.0)

    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end


def test_a_segment_longer_than_the_window_stands_alone() -> None:
    source = transcript([(0.0, 300.0, "очень длинный фрагмент"), (300.0, 310.0, "короткий")])
    chunks = chunk_transcript(source, window_s=90.0, overlap_s=15.0)

    assert chunks[0].text == "очень длинный фрагмент"
    assert chunks[0].end == 300.0


def test_no_chunks_from_an_empty_transcript() -> None:
    assert chunk_transcript(transcript([]), window_s=90.0, overlap_s=15.0) == []
