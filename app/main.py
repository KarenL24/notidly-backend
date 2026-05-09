from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import traceback

from .pipeline.preprocess import load_and_preprocess
from .pipeline.pitch import estimate_pitch
from .pipeline.segment import segment_notes
from .pipeline.quantize import full_rhythm_analysis
from .pipeline.score import build_score, score_to_musicxml_bytes, score_to_pdf_bytes

app = FastAPI(title="Notidly Transcription API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoteResponse(BaseModel):
    start_beat: float
    duration_beats: float
    midi_pitch: int
    freq_hz: float
    confidence: float


class PitchFrameResponse(BaseModel):
    time_sec: float
    freq_hz: float
    confidence: float
    voiced: bool


class RhythmResponse(BaseModel):
    tempo_bpm: float
    time_signature_numerator: int
    time_signature_denominator: int


class TranscribeResponse(BaseModel):
    duration_sec: float
    note_count: int
    rhythm: RhythmResponse
    notes: list[NoteResponse]
    pitch_frames: list[PitchFrameResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    if file.content_type not in ("audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a WAV file.")

    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File is too small to be a valid WAV.")

    try:
        audio = load_and_preprocess(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode audio. Ensure the file is a mono WAV.")

    try:
        pitch_frames, note_events = estimate_pitch(audio)
        raw_notes = segment_notes(note_events)
        quantized_notes, rhythm = full_rhythm_analysis(raw_notes, audio.samples, audio.sr)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Transcription failed. See server logs.")

    return TranscribeResponse(
        duration_sec=audio.duration_sec,
        note_count=len(quantized_notes),
        rhythm=RhythmResponse(
            tempo_bpm=rhythm.tempo_bpm,
            time_signature_numerator=rhythm.time_signature[0],
            time_signature_denominator=rhythm.time_signature[1],
        ),
        notes=[NoteResponse(**n.__dict__) for n in quantized_notes],
        pitch_frames=[PitchFrameResponse(**f.__dict__) for f in pitch_frames],
    )


@app.post("/musicxml")
async def get_musicxml(file: UploadFile = File(...)):
    """Run full pipeline and return a MusicXML file ready for sheet music rendering."""
    if file.content_type not in ("audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a WAV file.")

    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File is too small.")

    try:
        audio = load_and_preprocess(raw_bytes)
        pitch_frames, note_events = estimate_pitch(audio)
        raw_notes = segment_notes(note_events)
        quantized_notes, rhythm = full_rhythm_analysis(raw_notes, audio.samples, audio.sr)
        score = build_score(quantized_notes, rhythm)
        xml_bytes = score_to_musicxml_bytes(score)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Score generation failed. See server logs.")

    return Response(
        content=xml_bytes,
        media_type="application/vnd.recordare.musicxml+xml",
        headers={"Content-Disposition": "attachment; filename=transcription.xml"},
    )


@app.post("/pdf")
async def get_pdf(file: UploadFile = File(...)):
    """Run full pipeline and return a rendered PDF sheet music file."""
    if file.content_type not in ("audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a WAV file.")

    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File is too small.")

    try:
        audio = load_and_preprocess(raw_bytes)
        pitch_frames, note_events = estimate_pitch(audio)
        raw_notes = segment_notes(note_events)
        quantized_notes, rhythm = full_rhythm_analysis(raw_notes, audio.samples, audio.sr)
        score = build_score(quantized_notes, rhythm)
        pdf_bytes = score_to_pdf_bytes(score)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="PDF generation failed. See server logs.")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=transcription.pdf"},
    )
