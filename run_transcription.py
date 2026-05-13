"""Quick terminal test: run a WAV through the transcription pipeline."""
import sys
from pathlib import Path

from app.pipeline.preprocess import load_and_preprocess
from app.pipeline.pitch import estimate_pitch, freq_to_midi
from app.pipeline.segment import segment_notes

wav_path = sys.argv[1] if len(sys.argv) > 1 else "New Recording 3.wav"

print(f"Loading {wav_path}...")
audio = load_and_preprocess(Path(wav_path).read_bytes())
print(f"  sample rate: {audio.sr} Hz, duration: {audio.duration_sec:.2f}s")

print("Running pitch estimation (basic-pitch)...")
pitch_frames, note_events = estimate_pitch(audio)
print(f"  {len(pitch_frames)} pitch frames, {len(note_events)} raw note events")

print("Segmenting notes...")
notes = segment_notes(note_events)
print(f"  {len(notes)} notes after filtering/merging\n")

print(f"{'#':<4} {'start':>7} {'end':>7} {'dur':>6}  {'MIDI':>4}  note")
print("-" * 42)
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
for i, n in enumerate(notes):
    name = NOTE_NAMES[n.midi_pitch % 12] + str(n.midi_pitch // 12 - 1)
    print(f"{i+1:<4} {n.start_sec:>6.2f}s {n.end_sec:>6.2f}s {n.duration_sec:>5.2f}s  {n.midi_pitch:>4}  {name}")
