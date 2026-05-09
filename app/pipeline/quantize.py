import numpy as np
import librosa
from dataclasses import dataclass
from .segment import RawNote

SUPPORTED_METERS = [(4, 4), (3, 4), (6, 8)]
GRID_DIVISIONS = [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]  # double whole → eighth


@dataclass
class QuantizedNote:
    start_beat: float        # beat position (quarter notes from bar 1 beat 1)
    duration_beats: float    # duration in quarter notes
    midi_pitch: int
    freq_hz: float
    confidence: float


@dataclass
class RhythmInfo:
    tempo_bpm: float
    time_signature: tuple[int, int]  # (numerator, denominator)
    beats_per_bar: float             # in quarter notes


def detect_tempo(audio_samples: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=audio_samples, sr=sr)
    # beat_track returns an array in newer librosa — take scalar
    bpm = float(np.atleast_1d(tempo)[0])
    # Clamp to a musical range
    if bpm < 40:
        bpm *= 2
    elif bpm > 220:
        bpm /= 2
    return round(bpm, 1)


def detect_time_signature(notes: list[RawNote], tempo_bpm: float) -> tuple[int, int]:
    """Heuristic: count note onsets per bar and pick best-fitting meter."""
    if not notes or tempo_bpm <= 0:
        return (4, 4)

    beat_sec = 60.0 / tempo_bpm
    bar_4_4 = beat_sec * 4
    bar_3_4 = beat_sec * 3

    onsets = np.array([n.start_sec for n in notes])
    total_dur = notes[-1].end_sec if notes else 1.0

    score_4_4 = _meter_fit_score(onsets, bar_4_4, total_dur)
    score_3_4 = _meter_fit_score(onsets, bar_3_4, total_dur)

    return (4, 4) if score_4_4 >= score_3_4 else (3, 4)


def quantize_notes(
    notes: list[RawNote],
    tempo_bpm: float,
    time_sig: tuple[int, int],
) -> list[QuantizedNote]:
    """Snap note start times and durations to the nearest rhythmic grid."""
    if not notes:
        return []

    beat_sec = 60.0 / tempo_bpm
    quantized = []

    for note in notes:
        start_beat = note.start_sec / beat_sec
        dur_beats = note.duration_sec / beat_sec

        q_start = _snap_to_grid(start_beat, GRID_DIVISIONS)
        q_dur = _snap_to_grid(dur_beats, GRID_DIVISIONS)
        q_dur = max(q_dur, GRID_DIVISIONS[-1])  # minimum one grid unit

        quantized.append(QuantizedNote(
            start_beat=q_start,
            duration_beats=q_dur,
            midi_pitch=note.midi_pitch,
            freq_hz=note.freq_hz,
            confidence=note.confidence,
        ))

    return quantized


def full_rhythm_analysis(
    notes: list[RawNote],
    audio_samples: np.ndarray,
    sr: int,
) -> tuple[list[QuantizedNote], RhythmInfo]:
    """Run tempo detection, time signature detection, and quantization in one call."""
    tempo = detect_tempo(audio_samples, sr)
    time_sig = detect_time_signature(notes, tempo)
    beats_per_bar = time_sig[0] * (4 / time_sig[1])

    quantized = quantize_notes(notes, tempo, time_sig)
    rhythm = RhythmInfo(
        tempo_bpm=tempo,
        time_signature=time_sig,
        beats_per_bar=beats_per_bar,
    )
    return quantized, rhythm


def _snap_to_grid(value: float, grid: list[float]) -> float:
    """Round value to nearest grid division."""
    best = grid[0]
    best_err = abs(round(value / grid[0]) * grid[0] - value)
    for div in grid:
        snapped = round(value / div) * div
        err = abs(snapped - value)
        if err < best_err:
            best_err = err
            best = snapped
    return max(best, grid[-1])


def _meter_fit_score(onsets: np.ndarray, bar_sec: float, total_dur: float) -> float:
    """Score how well onsets align to a bar grid (higher = better fit)."""
    if bar_sec <= 0:
        return 0.0
    phases = onsets % bar_sec
    # Notes near bar boundaries score higher
    score = float(np.mean(np.cos(2 * np.pi * phases / bar_sec)))
    return score
