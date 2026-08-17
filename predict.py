"""
predict.py  —  inference on a single audio file
Public API (imported by app.py):
    analyze(audio_file)  -> dict   full pipeline, CLAUDE.md contract
    predict_cycle(y)     -> (label, confidence, probs_dict)
Constants re-exported for app.py:
    SR, CYCLE_SECS, CYCLE_SAMPLES, N_MELS, N_FFT, HOP_LENGTH, CLASSES
"""

from pathlib import Path
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ── constants (must match prep.py exactly) ───────────────────────────────────
SR            = 4000
CYCLE_SECS    = 5.0
CYCLE_SAMPLES = int(SR * CYCLE_SECS)   # 20 000
N_MELS        = 64
N_FFT         = 512
HOP_LENGTH    = 128
CLASSES       = ["both", "crackle", "normal", "wheeze"]

MIN_DURATION  = 2.0
SILENCE_THR   = 0.005
CONF_THR      = 0.5

_MAGMA = plt.cm.magma  # colormap used by prep.py imsave

# transform matches what ImageFolder applies (RGBA -> RGB was for PNG loading;
# here we build an RGB PIL image directly, so no convert step needed)
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None


# ── model loading ─────────────────────────────────────────────────────────────
def _load_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 4)
    for path in [Path("model2.pth"), Path("model.pth")]:
        if path.exists():
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            break
    else:
        raise FileNotFoundError("No model weights found. Run train2.py or train.py first.")
    m.eval()
    _model = m
    return _model


# ── audio helpers ─────────────────────────────────────────────────────────────
def _pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def _audio_to_tensor(y: np.ndarray) -> torch.Tensor:
    """
    Convert a normalised audio segment to a model-ready tensor.
    Replicates prep.py exactly: mel -> power_to_db -> flipud -> normalise
    to [0,1] -> magma colormap -> RGB PIL image -> ToTensor -> Normalize.
    """
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = np.flipud(mel_db)

    # Normalise to [0,1] as matplotlib's imsave does internally
    lo, hi  = img.min(), img.max()
    img_01  = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    # Apply magma colormap -> RGBA [0,1] -> RGB uint8
    rgba = _MAGMA(img_01)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return _TRANSFORM(Image.fromarray(rgb, mode="RGB")).unsqueeze(0)  # [1,3,H,W]


def _infer(tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(_load_model()(tensor), dim=1)[0].numpy()


# ── public: single-cycle prediction ──────────────────────────────────────────
def predict_cycle(y: np.ndarray) -> tuple:
    """
    Run model on one audio segment (numpy float32, already normalised, SR=4000).
    Pads/trims to CYCLE_SAMPLES internally.
    Returns (label, confidence, {class: probability}).
    """
    cycle = _pad_or_trim(y, CYCLE_SAMPLES)
    probs = _infer(_audio_to_tensor(cycle))
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASSES, probs)}


# ── public: full analyze pipeline (CLAUDE.md contract) ───────────────────────
def analyze(audio_file) -> dict:
    """
    Load audio_file, quality-check, predict, find salient region.
    Returns the dict specified in CLAUDE.md; on failure returns {"error": str}.
    """
    # ── load & validate ──────────────────────────────────────────────────
    try:
        y, _ = librosa.load(audio_file, sr=SR)
    except Exception as exc:
        return {"error": f"Could not load audio: {exc}"}

    duration = len(y) / SR

    if duration < MIN_DURATION:
        return {"error": f"Audio too short ({duration:.1f}s — need ≥{MIN_DURATION}s)."}

    peak = np.max(np.abs(y))
    if peak < SILENCE_THR:
        return {"error": "Audio appears silent. Check your microphone and re-record."}

    y = y / peak   # normalise once; all downstream slices share this

    # ── fixed 5s windows (fallback segmentation for uploaded audio) ──────
    n_cycles   = max(1, int(np.floor(len(y) / CYCLE_SAMPLES)))
    cycles_out = []
    all_probs  = []

    for i in range(n_cycles):
        t0 = i * CYCLE_SECS
        t1 = min(t0 + CYCLE_SECS, duration)
        s0, s1 = int(t0 * SR), int(t1 * SR)

        label, conf, pd = predict_cycle(y[s0:s1])
        cycles_out.append((t0, t1, label))
        all_probs.append([pd[c] for c in CLASSES])

    mean_probs = np.mean(all_probs, axis=0)
    top_idx    = int(np.argmax(mean_probs))
    label      = CLASSES[top_idx]
    confidence = float(mean_probs[top_idx])

    if confidence < CONF_THR:
        return {"error": f"Signal quality too low (confidence={confidence:.2f}). Re-record."}

    quality = "good" if confidence >= 0.7 else "poor"

    # ── salient region: mute 1s chunks, find largest confidence drop ─────
    probe   = _pad_or_trim(y, CYCLE_SAMPLES)
    base_c  = float(_infer(_audio_to_tensor(probe))[top_idx])
    chunk_n = SR   # 1 second of samples

    best_drop, salient = -1.0, (0.0, 1.0)
    for k in range(int(CYCLE_SECS)):
        muted = probe.copy()
        muted[k * chunk_n : (k + 1) * chunk_n] = 0.0
        drop  = base_c - float(_infer(_audio_to_tensor(muted))[top_idx])
        if drop > best_drop:
            best_drop = drop
            salient   = (float(k), float(k + 1))

    # ── mel spectrogram of first 5s for display ──────────────────────────
    mel  = librosa.feature.melspectrogram(y=probe, sr=SR, n_mels=N_MELS,
                                           n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec = librosa.power_to_db(mel, ref=np.max)

    return {
        "label":      label,
        "confidence": confidence,
        "all_probs":  {c: float(p) for c, p in zip(CLASSES, mean_probs)},
        "quality":    quality,
        "cycles":     cycles_out,
        "salient":    salient,
        "spec":       spec,
    }
