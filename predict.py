"""
predict.py  —  analyze() contract for app.py

STUB: label / confidence / all_probs / cycles / salient are hardcoded
plausible values. No model.pth is loaded yet — this exists so app.py
can be built against a stable interface. Audio loading, the quality
gate, and the spectrogram are real (librosa), since those don't need
a trained model and app.py needs something real to display.

Usage:
  .venv\\Scripts\\python.exe predict.py path\\to\\file.wav
"""

import sys
from pathlib import Path

import numpy as np
import librosa

# ── constants — MUST match prep.py exactly ─────────────────────────────────
SR          = 4000
N_MELS      = 64
N_FFT       = 512
HOP_LENGTH  = 128

MIN_DURATION_SECS   = 2.0
MIN_MEAN_ABS_AMP    = 0.005
MIN_TOP_CONFIDENCE  = 0.5

CLASSES = ["normal", "crackle", "wheeze", "both"]


def _empty_result(quality: str, message: str) -> dict:
    return {
        "label": "normal",
        "confidence": 0.0,
        "all_probs": {c: 0.25 for c in CLASSES},
        "quality": quality,
        "cycles": [],
        "salient": None,
        "spec": None,
        "message": message,
    }


def analyze(audio_file) -> dict:
    """
    Classify a respiratory audio file. Never raises — unusable audio
    is reported back via quality="poor" + a message, not an exception.
    """
    # ── load ─────────────────────────────────────────────────────────────
    try:
        y, _ = librosa.load(audio_file, sr=SR)
    except Exception as exc:
        return _empty_result("poor", f"could not read audio file: {exc}")

    if len(y) == 0:
        return _empty_result("poor", "audio file is empty")

    duration = len(y) / SR

    # ── graceful refusal gate ───────────────────────────────────────────
    if duration < MIN_DURATION_SECS:
        return _empty_result(
            "poor", f"recording too short ({duration:.1f}s) - need at least "
                    f"{MIN_DURATION_SECS:.0f}s"
        )

    mean_abs_amp = float(np.mean(np.abs(y)))
    if mean_abs_amp < MIN_MEAN_ABS_AMP:
        return _empty_result("poor", "signal too quiet - looks like silence, re-record")

    # ── normalize (same as prep.py) ─────────────────────────────────────
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # ── real mel spectrogram, same params as prep.py ────────────────────
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    spec = np.flipud(mel_db)

    # ── HARDCODED classification (no model loaded yet) ─────────────────
    label = "crackle"
    all_probs = {
        "normal": 0.12,
        "crackle": 0.61,
        "wheeze": 0.19,
        "both": 0.08,
    }
    confidence = all_probs[label]

    if confidence < MIN_TOP_CONFIDENCE:
        return _empty_result("poor", "signal quality too low, re-record")

    # ── hardcoded breath-cycle timeline, spaced across the real duration ─
    cycle_labels = ["normal", "crackle", "crackle", "normal"]
    n_cycles = max(1, min(4, int(duration // 2.5) or 1))
    edges = np.linspace(0, duration, n_cycles + 1)
    cycles = [
        (float(edges[i]), float(edges[i + 1]), cycle_labels[i % len(cycle_labels)])
        for i in range(n_cycles)
    ]

    # ── hardcoded salient region — middle third of the recording ────────
    salient = (float(duration / 3), float(2 * duration / 3))

    return {
        "label": label,
        "confidence": confidence,
        "all_probs": all_probs,
        "quality": "good",
        "cycles": cycles,
        "salient": salient,
        "spec": spec,
        "message": None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: predict.py path\\to\\file.wav")
        sys.exit(1)

    path = Path(sys.argv[1])
    result = analyze(path)

    print(f"file:       {path.name}")
    print(f"quality:    {result['quality']}")
    if result["message"]:
        print(f"message:    {result['message']}")
    print(f"label:      {result['label']}")
    print(f"confidence: {result['confidence']:.2f}")
    print(f"all_probs:  {result['all_probs']}")
    print(f"cycles:     {result['cycles']}")
    print(f"salient:    {result['salient']}")
    spec = result["spec"]
    print(f"spec shape: {spec.shape if spec is not None else None}")
