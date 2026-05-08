"""
Smoke tests — generate synthetic sine waves and run the full pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""
import io
import numpy as np
import soundfile as sf

from app.pipeline.preprocess import load_and_preprocess
from app.pipeline.pitch import estimate_pitch, freq_to_midi
from app.pipeline.segment import segment_notes


def _make_sine_wav(freq_hz: float = 440.0, duration_sec: float = 1.5, sr: int = 44100) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def test_preprocess_resamples():
    audio = load_and_preprocess(_make_sine_wav(sr=44100))
    assert audio.sr == 16000
    assert audio.samples.dtype == np.float32
    assert abs(audio.duration_sec - 1.5) < 0.1


def test_preprocess_normalizes():
    audio = load_and_preprocess(_make_sine_wav())
    assert np.abs(audio.samples).max() <= 1.0 + 1e-5


def test_freq_to_midi():
    assert freq_to_midi(440.0) == 69
    assert freq_to_midi(261.63) == 60
    assert freq_to_midi(0.0) is None


def test_pitch_returns_frames_and_events():
    audio = load_and_preprocess(_make_sine_wav(freq_hz=440.0, duration_sec=1.0))
    frames, note_events = estimate_pitch(audio)
    assert len(frames) > 50
    assert isinstance(note_events, list)


def test_segment_finds_a4():
    """A clean 440 Hz sine should produce MIDI 69 (A4)."""
    audio = load_and_preprocess(_make_sine_wav(freq_hz=440.0, duration_sec=1.5))
    _, note_events = estimate_pitch(audio)
    notes = segment_notes(note_events)
    assert len(notes) >= 1, "Expected at least one note"
    assert notes[0].midi_pitch == 69, f"Expected MIDI 69 (A4), got {notes[0].midi_pitch}"
