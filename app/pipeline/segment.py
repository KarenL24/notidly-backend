import numpy as np
from dataclasses import dataclass

MIDI_MIN = 48   # C3 — reasonable low vocal limit
MIDI_MAX = 84   # C6 — reasonable high vocal limit
MERGE_GAP_SEC = 0.15       # merge same-pitch notes within 150ms
SEMITONE_MERGE_THRESH = 2  # merge notes within 2 semitones (handles vibrato drift)
MIN_DURATION_SEC = 0.15    # drop notes shorter than 150ms after merging


@dataclass
class RawNote:
    start_sec: float
    end_sec: float
    duration_sec: float
    midi_pitch: int
    freq_hz: float
    confidence: float


def segment_notes(note_events: list, *_) -> list[RawNote]:
    """
    Convert basic-pitch note_events to RawNote list.
    note_events items: (start_sec, end_sec, midi_pitch, amplitude, pitch_bend)
    """
    notes = []
    for event in note_events:
        start_sec, end_sec, midi_pitch, amplitude, *_ = event
        midi_pitch = int(round(midi_pitch))
        if not (MIDI_MIN <= midi_pitch <= MIDI_MAX):
            continue
        freq_hz = 440.0 * 2 ** ((midi_pitch - 69) / 12.0)
        notes.append(RawNote(
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration_sec=float(end_sec - start_sec),
            midi_pitch=midi_pitch,
            freq_hz=freq_hz,
            confidence=float(amplitude),
        ))

    notes = sorted(notes, key=lambda n: n.start_sec)
    notes = _remove_octave_errors(notes)
    notes = _merge_nearby(notes)
    notes = [n for n in notes if n.duration_sec >= MIN_DURATION_SEC]
    return notes


def _remove_octave_errors(notes: list[RawNote]) -> list[RawNote]:
    """Remove notes that are more than an octave away from the local median pitch."""
    if len(notes) < 3:
        return notes
    pitches = [n.midi_pitch for n in notes]
    median = float(np.median(pitches))
    return [n for n in notes if abs(n.midi_pitch - median) <= 12]


def _merge_nearby(notes: list[RawNote]) -> list[RawNote]:
    """Merge consecutive notes that are close in pitch and time (vibrato fragments)."""
    if not notes:
        return notes
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start_sec - prev.end_sec
        pitch_diff = abs(note.midi_pitch - prev.midi_pitch)
        if gap <= MERGE_GAP_SEC and pitch_diff <= SEMITONE_MERGE_THRESH:
            # Merge: keep the higher-confidence pitch, extend the duration
            dominant = note if note.confidence > prev.confidence else prev
            merged[-1] = RawNote(
                start_sec=prev.start_sec,
                end_sec=note.end_sec,
                duration_sec=note.end_sec - prev.start_sec,
                midi_pitch=dominant.midi_pitch,
                freq_hz=dominant.freq_hz,
                confidence=max(prev.confidence, note.confidence),
            )
        else:
            merged.append(note)
    return merged
