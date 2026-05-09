import numpy as np
import soundfile as sf
import tempfile
import os
from dataclasses import dataclass
from typing import Optional
from .preprocess import AudioData

_CONTOUR_HOP_SEC = 256 / 22050
_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class PitchFrame:
    time_sec: float
    freq_hz: float
    confidence: float
    voiced: bool


def estimate_pitch(audio: AudioData, **kwargs) -> tuple[list[PitchFrame], list]:
    """
    Run basic-pitch on preprocessed audio.
    Returns (pitch_frames, note_events).
    note_events: list of (start_sec, end_sec, midi_pitch, amplitude, pitch_bend)
    """
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio.samples, audio.sr)
        tmp_path = tmp.name

    try:
        model_output, _, note_events = predict(
            tmp_path,
            ICASSP_2022_MODEL_PATH,
            minimum_note_length=200,   # 200ms — prevents vibrato fragmentation
            minimum_frequency=80.0,
            maximum_frequency=800.0,   # cuts out octave-error harmonics above soprano range
            melodia_trick=True,
        )
    finally:
        os.unlink(tmp_path)

    pitch_frames = _contour_to_frames(model_output["contour"])
    return pitch_frames, note_events


def freq_to_midi(freq_hz: float) -> Optional[int]:
    if freq_hz <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq_hz / 440.0)))


def _contour_to_frames(contour: np.ndarray) -> list[PitchFrame]:
    frames = []
    for i, bins in enumerate(contour):
        confidence = float(bins.max())
        if confidence >= _CONFIDENCE_THRESHOLD:
            bin_idx = int(bins.argmax())
            midi_float = 21.0 + bin_idx / 3.0
            freq = 440.0 * 2 ** ((midi_float - 69.0) / 12.0)
            voiced = True
        else:
            freq = 0.0
            voiced = False
        frames.append(PitchFrame(
            time_sec=i * _CONTOUR_HOP_SEC,
            freq_hz=freq,
            confidence=confidence,
            voiced=voiced,
        ))
    return frames
