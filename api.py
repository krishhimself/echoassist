# api.py - EchoAssist FastAPI backend
# Run: .venv/Scripts/uvicorn.exe api:app --reload --port 8000
import io
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from predict import analyze, SR

app = FastAPI(title="EchoAssist API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES_DIR = Path("samples")


def _safe_result(result: dict) -> dict:
    """Convert numpy arrays to JSON-serialisable Python lists."""
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyse an uploaded audio file. Returns full prediction dict."""
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = analyze(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return JSONResponse(_safe_result(result))


@app.get("/api/analyze/sample/{name}")
async def analyze_sample(name: str):
    """Analyse one of the curated gallery samples by filename."""
    wav = SAMPLES_DIR / name
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
    result = analyze(str(wav))
    return JSONResponse(_safe_result(result))


@app.get("/api/salient-audio/{name}")
async def salient_audio(name: str, t0: float = 0.0, t1: float = 1.0):
    """Return the salient 1-second audio segment as a WAV stream."""
    wav = SAMPLES_DIR / name
    if not wav.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    y, _ = librosa.load(str(wav), sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    s0, s1 = int(t0 * SR), int(t1 * SR)
    segment = y[s0:s1]
    if len(segment) == 0:
        segment = y[:SR]
    buf = io.BytesIO()
    sf.write(buf, segment, SR, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")


@app.get("/api/samples")
async def list_samples():
    """List available curated sample files."""
    SAMPLE_META = [
        {"label": "Normal — patient 210",  "file": "normal_patient210_Al.wav",  "class": "normal"},
        {"label": "Normal — patient 112",  "file": "normal_patient112_Ar.wav",  "class": "normal"},
        {"label": "Crackle — patient 223", "file": "crackle_patient223_Lr.wav", "class": "crackle"},
        {"label": "Crackle — patient 205", "file": "crackle_patient205_Ar.wav", "class": "crackle"},
        {"label": "Wheeze — patient 223",  "file": "wheeze_patient223_Ar.wav",  "class": "wheeze"},
        {"label": "Wheeze — patient 206",  "file": "wheeze_patient206_Pl.wav",  "class": "wheeze"},
        {"label": "Both — patient 156",    "file": "both_patient156_Pr.wav",    "class": "both"},
        {"label": "Silent (edge case)",    "file": "silent.wav",               "class": None},
        {"label": "Too short (edge case)", "file": "short.wav",                "class": None},
    ]
    available = [s for s in SAMPLE_META if (SAMPLES_DIR / s["file"]).exists()]
    return available


@app.get("/api/report/{name}")
async def get_clinical_report(name: str):
    """Generate and return clinical HTML report for a sample file."""
    from report_generator import generate_report
    from fastapi.responses import FileResponse
    
    wav = SAMPLES_DIR / name
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
        
    result = analyze(str(wav))
    if "error" in result and result.get("quality") == "rejected":
        # Report can still be generated for failed quality gates to show details
        pass
    elif "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    report_path = generate_report(result, str(wav))
    return FileResponse(
        path=report_path, 
        filename=Path(report_path).name, 
        media_type="text/html"
    )


# Serve frontend static files — mount last so API routes take priority
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
