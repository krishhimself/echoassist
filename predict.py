"""
predict.py  —  analyze() contract for app.py

Loads model.pth (ResNet18, trained on 5-second-cycle mel spectrograms —
see prep.py / train.py) and runs real inference. The audio is chopped
into 5-second windows (same CYCLE_SECS as prep.py), each windowed
spectrogram is rendered through matplotlib's magma colormap exactly the
way prep.py wrote the training PNGs (per-window min/max normalization),
then classified. Per-window predictions become the cycle timeline;
their mean becomes the overall label/confidence.

Explainability (salient region) is occlusion-based: the audio is split
into 1-second chunks, each is muted in turn, the whole pipeline is
re-run, and the chunk whose removal causes the largest confidence drop
for the predicted class is reported as salient. A plain loop, no
Grad-CAM.

Usage:
  .venv\\Scripts\\python.exe predict.py path\\to\\file.wav
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import librosa
import matplotlib
import matplotlib.colors as mcolors
import torch
from torchvision import models, transforms

# ── constants — MUST match prep.py / train.py exactly ───────────────────────
SR            = 4000
N_MELS        = 64
N_FFT         = 512
HOP_LENGTH    = 128
CYCLE_SECS    = 5.0
CYCLE_SAMPLES = int(SR * CYCLE_SECS)  # 20 000 samples, matches prep.py

MIN_DURATION_SECS   = 2.0
MIN_MEAN_ABS_AMP    = 0.005
MIN_TOP_CONFIDENCE  = 0.5

OCCLUSION_CHUNK_SECS = 1.0  # explainability: mute-one-second-at-a-time

MODEL_PATH = Path("model.pth")

# ImageFolder sorts class dirs alphabetically — this is the model's output order
MODEL_CLASSES = ["both", "crackle", "normal", "wheeze"]
# display / dict order used throughout the rest of the app
CLASSES = ["normal", "crackle", "wheeze", "both"]

_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


# ── model loading (once, cached) ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(MODEL_CLASSES))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


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


def _pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def _mel_db_to_rgb(mel_db_flipped: np.ndarray) -> np.ndarray:
    """Reproduce mpimg.imsave(..., cmap='magma') exactly: per-array min/max
    normalization through the magma colormap, dropping alpha. This is the
    same pixel data the model was trained on."""
    norm = mcolors.Normalize(vmin=mel_db_flipped.min(), vmax=mel_db_flipped.max())
    rgba = matplotlib.colormaps["magma"](norm(mel_db_flipped))
    return (rgba[..., :3] * 255).astype(np.uint8)


def _window_spec(cycle_audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=cycle_audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.flipud(mel_db)


def _predict_probs(model, y: np.ndarray, window_bounds: list) -> np.ndarray:
    """Run the model over each 5s window of y. Returns (n_windows, 4) softmax
    probabilities in MODEL_CLASSES order."""
    specs = []
    for w0, _w1 in window_bounds:
        s0 = int(w0 * SR)
        s1 = s0 + CYCLE_SAMPLES
        window_audio = _pad_or_trim(y[s0:s1], CYCLE_SAMPLES)
        specs.append(_mel_db_to_rgb(_window_spec(window_audio)))

    batch = torch.stack([_transform(_to_pil(s)) for s in specs])
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def analyze(audio_file) -> dict:
    """
    Classify a respiratory audio file. Never raises — unusable audio
    or a missing model is reported back via quality="poor" + a message.
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

    try:
        model = _load_model()
    except Exception as exc:
        return _empty_result("poor", f"model unavailable: {exc}")

    # ── normalize (same as prep.py) ─────────────────────────────────────
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # ── full-clip mel spectrogram, only for display in app.py ──────────
    spec = _window_spec(y)

    # ── chop into 5-second windows (proxy for breath cycles) and classify ─
    window_starts = np.arange(0.0, duration, CYCLE_SECS)
    window_bounds = [
        (float(w_start), float(min(w_start + CYCLE_SECS, duration)))
        for w_start in window_starts
    ]

    probs = _predict_probs(model, y, window_bounds)  # (n_windows, 4), MODEL_CLASSES order

    # ── per-window cycle labels ──────────────────────────────────────────
    cycles = []
    for (w0, w1), p in zip(window_bounds, probs):
        cyc_label = MODEL_CLASSES[int(np.argmax(p))]
        cycles.append((w0, w1, cyc_label))

    # ── overall prediction = mean probability across windows ────────────
    mean_probs = probs.mean(axis=0)
    all_probs = {cls: float(mean_probs[i]) for i, cls in enumerate(MODEL_CLASSES)}
    label = max(all_probs, key=all_probs.get)
    confidence = all_probs[label]

    if confidence < MIN_TOP_CONFIDENCE:
        return _empty_result("poor", "signal quality too low, re-record")

    # ── occlusion explainability: mute 1s chunks, find the biggest drop ──
    label_idx = MODEL_CLASSES.index(label)
    n_chunks = int(np.ceil(duration / OCCLUSION_CHUNK_SECS))
    salient = (0.0, min(OCCLUSION_CHUNK_SECS, duration))
    best_drop = -np.inf
    for i in range(n_chunks):
        c0 = i * OCCLUSION_CHUNK_SECS
        c1 = min(c0 + OCCLUSION_CHUNK_SECS, duration)
        y_muted = y.copy()
        y_muted[int(c0 * SR):int(c1 * SR)] = 0.0

        muted_probs = _predict_probs(model, y_muted, window_bounds).mean(axis=0)
        drop = confidence - float(muted_probs[label_idx])

        if drop > best_drop:
            best_drop = drop
            salient = (c0, c1)

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


def _to_pil(rgb_array: np.ndarray):
    from PIL import Image
    return Image.fromarray(rgb_array, mode="RGB")


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
