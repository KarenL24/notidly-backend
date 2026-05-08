from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from .pipeline.preprocess import load_and_preprocess
from .pipeline.pitch import estimate_pitch
from .pipeline.segment import segment_notes

app = FastAPI(title="Notidly Transcription API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NoteResponse(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float
    midi_pitch: int
    freq_hz: float
    confidence: float

class PitchFrameResponse(BaseModel):
    time_sec: float
    freq_hz: float
    confidence: float
    voiced: bool

class TranscribeResponse(BaseModel):
    duration_sec: float
    note_count: int
    notes: list[NoteResponse]
    pitch_frames: list[PitchFrameResponse]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...), model_capacity: str = "full"):
    if file.content_type not in ("audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Upload a WAV file.")
    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(status_code=400, detail="File too small.")
    try:
        audio = load_and_preprocess(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode audio.")
    try:
        pitch_frames = estimate_pitch(audio, model_capacity=model_capacity)
        notes = segment_notes(pitch_frames, audio)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Transcription failed.")
    return TranscribeResponse(
        duration_sec=audio.duration_sec, note_count=len(notes),
        notes=[NoteResponse(**n.__dict__) for n in notes],
        pitch_frames=[PitchFrameResponse(**f.__dict__) for f in pitch_frames],
    )
