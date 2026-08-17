"""
api.py  --  FastAPI wrapper around predict.analyze()
Endpoints:
    GET  /health     liveness + model status
    POST /analyze    multipart WAV upload -> JSON analysis result
Run:
    .venv\\Scripts\\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from predict import analyze

app = FastAPI(title="EchoAssist API", version="1.0")

_MODEL_FILE = next((p for p in ["model2.pth", "model.pth"] if Path(p).exists()), None)


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return [_serialise(v) for v in obj]
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _MODEL_FILE,
        "model_exists": _MODEL_FILE is not None and Path(_MODEL_FILE).exists(),
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = analyze(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return JSONResponse(content=_serialise(result))
