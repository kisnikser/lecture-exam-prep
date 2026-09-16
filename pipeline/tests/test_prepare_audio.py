from pathlib import Path
from typing import ClassVar

from examprep import download
from examprep.download import transcribe_source

SLUG = "hps-skvorchevsky"
VIDEO = "abc123"


def test_prefers_the_prepared_wav(monkeypatch, tmp_path) -> None:
    wav = tmp_path / f"{VIDEO}.wav"
    monkeypatch.setattr(download, "wav_path", lambda slug, video_id: wav)

    assert transcribe_source(SLUG, VIDEO).suffix == ".m4a"
    wav.write_bytes(b"pcm")
    assert transcribe_source(SLUG, VIDEO) == wav


def test_ffmpeg_gets_absolute_paths_for_dash_prefixed_ids(monkeypatch) -> None:
    """A video id starting with "-" must not reach ffmpeg as an option."""

    calls: list[list[str]] = []
    monkeypatch.setattr(download.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(download, "audio_path", lambda slug, video_id: Path(__file__))
    monkeypatch.setattr(download, "wav_path", lambda slug, video_id: Path(f"/tmp/{video_id}.wav"))

    class Source:
        video_id = "-SSdI8Rsj64"

    class Course:
        sources: ClassVar[list[Source]] = [Source()]

    monkeypatch.setattr(download, "load_course", lambda slug: Course())
    download.prepare_audio(SLUG)

    assert len(calls) == 1
    for argument in calls[0]:
        assert not argument.startswith("-S")
    assert "-ar" in calls[0]
    assert "16000" in calls[0]
