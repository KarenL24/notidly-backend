import io
import tempfile
import os
from music21 import stream, note, chord, tempo, meter, key, pitch as m21pitch
from .quantize import QuantizedNote, RhythmInfo

REST_THRESHOLD_BEATS = 0.125  # gaps larger than this insert a rest


def build_score(
    quantized_notes: list[QuantizedNote],
    rhythm: RhythmInfo,
) -> stream.Score:
    """Assemble a music21 Score from quantized notes + rhythm metadata."""
    s = stream.Score()
    part = stream.Part()

    # Header
    part.append(tempo.MetronomeMark(number=rhythm.tempo_bpm))
    num, den = rhythm.time_signature
    part.append(meter.TimeSignature(f"{num}/{den}"))

    # Detect key signature from MIDI pitches
    ks = _detect_key(quantized_notes)
    part.append(ks)

    if not quantized_notes:
        s.append(part)
        return s

    cursor = 0.0  # current beat position

    for qn in quantized_notes:
        gap = qn.start_beat - cursor
        if gap > REST_THRESHOLD_BEATS:
            rest = note.Rest(quarterLength=_clean_duration(gap))
            part.append(rest)

        n = note.Note()
        n.pitch = m21pitch.Pitch(midi=qn.midi_pitch)
        n.quarterLength = _clean_duration(qn.duration_beats)
        n.volume.velocity = int(min(127, max(1, qn.confidence * 127)))
        part.append(n)

        cursor = qn.start_beat + qn.duration_beats

    s.append(part)
    return s


def score_to_musicxml(score: stream.Score) -> str:
    """Return MusicXML as a string."""
    buf = io.StringIO()
    score.write("musicxml", fp=None)
    # music21 write to string
    xml_bytes = score.write("musicxml")
    with open(xml_bytes, "r") as f:
        return f.read()


def score_to_musicxml_bytes(score: stream.Score) -> bytes:
    """Return MusicXML as bytes, ready to send over HTTP."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        score.write("musicxml", fp=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _detect_key(notes: list[QuantizedNote]) -> key.Key:
    """Simple pitch-class histogram key detection."""
    if not notes:
        return key.Key("C")

    # Build pitch class histogram
    hist = [0] * 12
    for n in notes:
        hist[n.midi_pitch % 12] += 1

    # Krumhansl-Schmuckler profiles (major)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    best_corr = -999
    best_key = key.Key("C")

    for root in range(12):
        for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
            rotated = profile[root:] + profile[:root]
            corr = _correlation(hist, rotated)
            if corr > best_corr:
                best_corr = corr
                pitch_name = m21pitch.Pitch(root).name
                best_key = key.Key(pitch_name, mode)

    return best_key


def _correlation(a: list, b: list) -> float:
    a = [x - sum(a) / len(a) for x in a]
    b = [x - sum(b) / len(b) for x in b]
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x**2 for x in a) * sum(y**2 for y in b)) ** 0.5
    return num / den if den else 0.0


def _clean_duration(beats: float) -> float:
    """Clamp to valid music21 quarter lengths."""
    valid = [4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125]
    return min(valid, key=lambda v: abs(v - beats))
