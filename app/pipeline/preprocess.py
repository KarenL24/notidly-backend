import numpy as np
import librosa
import soundfile as sf
import io
from dataclasses import dataclass

CREPE_SR = 16000
MAX_DURATION_SEC = 120


@dataclass
class AudioData:
    samples: np.ndarray
    sr: int
    duration_sec: float


def load_and_preprocess(file_bytes: bytes) -> AudioData:
    buf = io.BytesIO(file_bytes)
    y, sr = librosa.load(buf, sr=None, mono=True)
    if len(y) / sr > MAX_DURATION_SEC:
        raise ValueError(f"Audio exceeds {MAX_DURATION_SEC}s limit")
    if sr != CREPE_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CREPE_SR)
        sr = CREPE_SR
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return AudioData(samples=y.astype(np.float32), sr=sr, duration_sec=len(y) / sr)


def compute_rms_envelope(audio: AudioData, frame_length: int = 512, hop_length: int = 160) -> np.ndarray:
    rms = librosa.feature.rms(y=audio.samples, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def silence_mask(rms: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    peak_rms = rms.max()
    if peak_rms == 0:
        return np.ones(len(rms), dtype=bool)
    db = 20 * np.log10(np.maximum(rms / peak_rms, 1e-10))
    return db < threshold_db
