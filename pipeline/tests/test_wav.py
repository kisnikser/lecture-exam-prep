import wave

import numpy as np

from examprep.transcribe.whisper import load_wav


def write_wav(path, channels: int = 1, rate: int = 16000, width: int = 2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(np.array([0, 16384, -16384], dtype="<i2").tobytes())


def test_reads_the_expected_layout(tmp_path) -> None:
    path = tmp_path / "a.wav"
    write_wav(path)

    audio = load_wav(path)
    assert audio is not None
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-6)


def test_returns_none_for_another_layout(tmp_path) -> None:
    stereo = tmp_path / "stereo.wav"
    write_wav(stereo, channels=2)
    assert load_wav(stereo) is None

    resampled = tmp_path / "44k.wav"
    write_wav(resampled, rate=44100)
    assert load_wav(resampled) is None


def test_returns_none_for_a_missing_file(tmp_path) -> None:
    assert load_wav(tmp_path / "нет.wav") is None
