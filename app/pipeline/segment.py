import numpy as np
import librosa
from dataclasses import dataclass
from .pitch import PitchFrame, freq_to_midi
from .preprocess import AudioData

MIN_NOTE_DURATION_SEC = 0.05
MIN_SILENCE_GAP_SEC = 0.04


@dataclass
class RawNote:
    start_sec: float
    end_sec: float
    duration_sec: float
    midi_pitch: int
    freq_hz: float
    confidence: float


def detect_onsets(audio: AudioData) -> np.ndarray:
    return librosa.onset.onset_detect(
        y=audio.samples, sr=audio.sr, units="time", hop_length=160, backtrack=True,
    )


def segment_notes(pitch_frames: list[PitchFrame], audio: AudioData) -> list[RawNote]:
    onsets = detect_onsets(audio)
    runs = _voiced_runs(pitch_frames)
    raw_notes = []
    for run_start, run_end in runs:
        run_frames = pitch_frames[run_start: run_end + 1]
        run_t_start = run_frames[0].time_sec
        run_t_end = run_frames[-1].time_sec
        split_times = sorted(set([run_t_start] + [t for t in onsets if run_t_start <= t <= run_t_end]))
        segments = list(zip(split_times, split_times[1:] + [run_t_end]))
        for seg_start, seg_end in segments:
            seg_frames = [f for f in run_frames if seg_start <= f.time_sec <= seg_end]
            if not seg_frames:
                continue
            duration = seg_end - seg_start
            if duration < MIN_NOTE_DURATION_SEC:
                continue
            freqs = [f.freq_hz for f in seg_frames if f.voiced]
            confs = [f.confidence for f in seg_frames if f.voiced]
            if not freqs:
                continue
            median_freq = float(np.median(freqs))
            midi = freq_to_midi(median_freq)
            if midi is None or midi < 36 or midi > 96:
                continue
            raw_notes.append(RawNote(
                start_sec=seg_start, end_sec=seg_end, duration_sec=duration,
                midi_pitch=midi, freq_hz=median_freq, confidence=float(np.median(confs)),
            ))
    return _merge_same_pitch_gaps(raw_notes)


def _voiced_runs(frames):
    runs, in_run, start = [], False, 0
    for i, f in enumerate(frames):
        if f.voiced and not in_run:
            in_run, start = True, i
        elif not f.voiced and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(frames) - 1))
    return runs


def _merge_same_pitch_gaps(notes):
    if not notes:
        return notes
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start_sec - prev.end_sec
        if note.midi_pitch == prev.midi_pitch and gap < MIN_SILENCE_GAP_SEC:
            merged[-1] = RawNote(
                start_sec=prev.start_sec, end_sec=note.end_sec,
                duration_sec=note.end_sec - prev.start_sec,
                midi_pitch=prev.midi_pitch,
                freq_hz=(prev.freq_hz + note.freq_hz) / 2,
                confidence=(prev.confidence + note.confidence) / 2,
            )
        else:
            merged.append(note)
    return merged
