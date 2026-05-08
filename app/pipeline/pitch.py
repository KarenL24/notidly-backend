import numpy as np
import crepe
from scipy.ndimage import median_filter
from dataclasses import dataclass
from typing import Optional
from .preprocess import AudioData, compute_rms_envelope, silence_mask

CONFIDENCE_THRESHOLD = 0.5
MEDIAN_FILTER_SIZE = 5


@dataclass
class PitchFrame:
    time_sec: float
    freq_hz: float
    confidence: float
    voiced: bool


def estimate_pitch(audio: AudioData, model_capacity: str = "full") -> list[PitchFrame]:
    times, freqs, confidences, _ = crepe.predict(
        audio.samples, audio.sr, viterbi=True, capacity=model_capacity, verbose=0,
    )
    freqs_filtered = median_filter(freqs.astype(np.float64), size=MEDIAN_FILTER_SIZE)
    rms = compute_rms_envelope(audio, hop_length=160)
    rms_aligned = _align_length(rms, len(times))
    silent = silence_mask(rms_aligned)
    frames = []
    for i, (t, f, c) in enumerate(zip(times, freqs_filtered, confidences)):
        voiced = bool(c >= CONFIDENCE_THRESHOLD and not silent[i] and f > 50.0)
        frames.append(PitchFrame(
            time_sec=float(t), freq_hz=float(f) if voiced else 0.0,
            confidence=float(c), voiced=voiced,
        ))
    return frames


def freq_to_midi(freq_hz: float) -> Optional[int]:
    if freq_hz <= 0:
        return None
    midi = 69 + 12 * np.log2(freq_hz / 440.0)
    return int(round(midi))


def _align_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    indices = np.round(np.linspace(0, len(arr) - 1, target_len)).astype(int)
    return arr[indices]
