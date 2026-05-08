from dataclasses import dataclass

MIDI_MIN = 36
MIDI_MAX = 96


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
    return sorted(notes, key=lambda n: n.start_sec)
