"""
predict.py  —  inference on a single audio file (Multi-Label Binary Classification + Grad-CAM)
Public API (imported by app.py):
    analyze(audio_file)  -> dict   full pipeline, CLAUDE.md contract
    predict_cycle(y)     -> (label, confidence, probs_dict)
    generate_gradcam(y)  -> ndarray (64x157 heatmap)
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
CONF_THR      = 0.15

_MAGMA = plt.cm.magma  # colormap used by prep.py imsave

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_model = None


# ── model loading (Multi-Label 2-Output Architecture) ─────────────────────────
def _load_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)  # Multi-label outputs: [has_crackle, has_wheeze]
    for path in [Path("model2.pth"), Path("model.pth")]:
        if path.exists():
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                if state_dict.get("fc.weight", torch.zeros(0)).shape[0] == 2:
                    m.load_state_dict(state_dict)
                else:
                    # Adapt old 4-class weights to 2-class multi-label outputs
                    old_w = state_dict.pop("fc.weight")
                    old_b = state_dict.pop("fc.bias")
                    m.fc.weight.data = torch.stack([old_w[1], old_w[3]])
                    m.fc.bias.data = torch.stack([old_b[1], old_b[3]])
                    m.load_state_dict(state_dict, strict=False)
            except Exception:
                pass
            break
    else:
        torch.save(m.state_dict(), "model.pth")
    m.eval()
    _model = m
    return _model


# ── audio helpers ─────────────────────────────────────────────────────────────
def _pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def _audio_to_tensor(y: np.ndarray) -> torch.Tensor:
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = np.flipud(mel_db)

    lo, hi  = img.min(), img.max()
    img_01  = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    rgba = _MAGMA(img_01)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return _TRANSFORM(Image.fromarray(rgb, mode="RGB")).unsqueeze(0)


def _infer(tensor: torch.Tensor) -> np.ndarray:
    """Returns derived 4-class probability vector [both, crackle, normal, wheeze] via multi-label sigmoids."""
    with torch.no_grad():
        logits = _load_model()(tensor)
        sig = torch.sigmoid(logits)[0].numpy()
        p_c, p_w = float(sig[0]), float(sig[1])

        p_both    = p_c * p_w
        p_crackle = p_c * (1.0 - p_w)
        p_wheeze  = (1.0 - p_c) * p_w
        p_normal  = (1.0 - p_c) * (1.0 - p_w)

        probs = np.array([p_both, p_crackle, p_normal, p_wheeze], dtype=np.float32)
        total = np.sum(probs)
        return probs / total if total > 0 else np.array([0.0, 0.0, 1.0, 0.0])


# ── Grad-CAM Heatmap Generator ────────────────────────────────────────────────
def generate_gradcam(y: np.ndarray, target_label: str | None = None) -> np.ndarray:
    """Generate 2D Grad-CAM activation heatmap array (64x157) using model's final conv layer."""
    m = _load_model()
    m.eval()

    probe = _pad_or_trim(y, CYCLE_SAMPLES)
    tensor = _audio_to_tensor(probe)
    tensor.requires_grad = True

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = m.layer4[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = m(tensor)  # logits for [has_crackle, has_wheeze]
    
    if target_label == "crackle":
        score = output[0, 0]
    elif target_label == "wheeze":
        score = output[0, 1]
    elif target_label == "both":
        score = output[0, 0] + output[0, 1]
    elif target_label == "normal":
        score = - (output[0, 0] + output[0, 1])
    else:
        top_idx = int(torch.argmax(output[0]))
        score = output[0, top_idx]

    m.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        return np.zeros((N_MELS, 157), dtype=np.float32)

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_img.resize((157, N_MELS), resample=Image.BILINEAR)
    return np.array(cam_resized, dtype=np.float32) / 255.0


# ── public: single-cycle prediction ──────────────────────────────────────────
def predict_cycle(y: np.ndarray) -> tuple:
    cycle = _pad_or_trim(y, CYCLE_SAMPLES)
    probs = _infer(_audio_to_tensor(cycle))
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), {c: float(p) for c, p in zip(CLASSES, probs)}


# ── public: full analyze pipeline (CLAUDE.md contract) ───────────────────────
def analyze(audio_file) -> dict:
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

    y = y / peak

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

    quality = "good" if confidence >= 0.30 else "poor"

    probe   = _pad_or_trim(y, CYCLE_SAMPLES)
    base_c  = float(_infer(_audio_to_tensor(probe))[top_idx])
    chunk_n = SR

    best_drop, salient = -1.0, (0.0, 1.0)
    for k in range(int(CYCLE_SECS)):
        muted = probe.copy()
        muted[k * chunk_n : (k + 1) * chunk_n] = 0.0
        drop  = base_c - float(_infer(_audio_to_tensor(muted))[top_idx])
        if drop > best_drop:
            best_drop = drop
            salient   = (float(k), float(k + 1))

    mel  = librosa.feature.melspectrogram(y=probe, sr=SR, n_mels=N_MELS,
                                           n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec = librosa.power_to_db(mel, ref=np.max)
    gradcam_map = generate_gradcam(probe, target_label=label)

    return {
        "label":      label,
        "confidence": confidence,
        "all_probs":  {c: float(p) for c, p in zip(CLASSES, mean_probs)},
        "quality":    quality,
        "cycles":     cycles_out,
        "salient":    salient,
        "spec":       spec,
        "gradcam":    gradcam_map,
    }
