"""Test the full pipeline through to PDF output."""
import sys
from pathlib import Path

from app.pipeline.preprocess import load_and_preprocess
from app.pipeline.pitch import estimate_pitch
from app.pipeline.segment import segment_notes
from app.pipeline.quantize import full_rhythm_analysis
from app.pipeline.score import build_score, score_to_pdf_bytes, score_to_musicxml_bytes

wav_path = sys.argv[1] if len(sys.argv) > 1 else "New Recording 3.wav"
out_pdf = sys.argv[2] if len(sys.argv) > 2 else "test_output.pdf"
out_xml = out_pdf.replace(".pdf", ".xml")

print(f"Loading {wav_path}...")
audio = load_and_preprocess(Path(wav_path).read_bytes())

print("Estimating pitch...")
_, note_events = estimate_pitch(audio)

print("Segmenting notes...")
notes = segment_notes(note_events)
print(f"  {len(notes)} notes")

print("Quantizing rhythm...")
quantized, rhythm = full_rhythm_analysis(notes, audio.samples, audio.sr)
print(f"  tempo={rhythm.tempo_bpm} bpm, time sig={rhythm.time_signature[0]}/{rhythm.time_signature[1]}")

print("Building score...")
score = build_score(quantized, rhythm)

print(f"Writing MusicXML -> {out_xml}")
Path(out_xml).write_bytes(score_to_musicxml_bytes(score))

print(f"Writing PDF -> {out_pdf} (requires LilyPond)")
try:
    Path(out_pdf).write_bytes(score_to_pdf_bytes(score))
    print("Done!")
except RuntimeError as e:
    print(f"PDF failed: {e}")
    print("MusicXML was still written — open it in MuseScore or Finale.")
