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

# Signal quality thresholds
CLIP_RATIO_THR = 0.01    # >1% of samples at max amplitude = clipping
RMS_MIN_THR    = 0.008   # RMS below this = too quiet
SNR_MIN_DB     = 6.0     # SNR below this = excessive noise

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


# ── signal quality checks ─────────────────────────────────────────────────────
def _check_signal_quality(y_raw: np.ndarray, sr: int = SR) -> tuple:
    """Run signal quality checks on raw (unnormalized) audio.
    Returns (passed: bool, reason: str).
    """
    # Clipping detection: fraction of samples at or near max amplitude
    abs_y = np.abs(y_raw)
    peak = np.max(abs_y) if len(abs_y) > 0 else 0.0
    if peak > 0:
        clip_ratio = np.mean(abs_y >= 0.99 * peak)
        if clip_ratio > CLIP_RATIO_THR:
            return False, f"Audio clipped ({clip_ratio:.1%} of samples at max amplitude). Reduce input gain and re-record."

    # RMS level check
    rms = float(np.sqrt(np.mean(y_raw ** 2)))
    if rms < RMS_MIN_THR:
        return False, f"Signal too quiet (RMS={rms:.4f}). Increase microphone sensitivity or move closer."

    # SNR estimation: compare signal RMS to noise floor
    # Noise floor estimated from bottom 10th percentile of short-time frame energies
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = frame_len // 2
    n_frames = max(1, (len(y_raw) - frame_len) // hop)
    frame_energies = np.array([
        np.sqrt(np.mean(y_raw[i * hop : i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])
    noise_floor = float(np.percentile(frame_energies, 10)) if len(frame_energies) > 0 else rms
    if noise_floor > 0:
        snr_db = 20.0 * np.log10(rms / noise_floor)
    else:
        snr_db = 60.0  # effectively clean
    if snr_db < SNR_MIN_DB:
        return False, f"Excessive background noise detected (SNR={snr_db:.1f} dB). Find a quieter environment and re-record."

    return True, "Passed all signal quality checks"


# ── public: full analyze pipeline (CLAUDE.md contract) ───────────────────────
def analyze(audio_file) -> dict:
    try:
        y, _ = librosa.load(audio_file, sr=SR)
    except Exception as exc:
        return {"error": f"Could not load audio: {exc}",
                "quality": "rejected", "quality_reason": f"Could not load audio: {exc}"}

    duration = len(y) / SR

    if duration < MIN_DURATION:
        return {"error": f"Audio too short ({duration:.1f}s — need ≥{MIN_DURATION}s).",
                "quality": "rejected",
                "quality_reason": f"Audio duration too short ({duration:.1f}s, minimum {MIN_DURATION}s required)"}

    peak = np.max(np.abs(y))
    if peak < SILENCE_THR:
        return {"error": "Audio appears silent. Check your microphone and re-record.",
                "quality": "rejected",
                "quality_reason": "Audio appears silent (peak amplitude below threshold). Check microphone connection."}

    # Advanced signal quality checks (on raw audio before normalization)
    sq_passed, sq_reason = _check_signal_quality(y)
    if not sq_passed:
        return {"error": sq_reason,
                "quality": "rejected", "quality_reason": sq_reason}

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
        return {"error": f"Signal quality too low (confidence={confidence:.2f}). Re-record.",
                "quality": "rejected",
                "quality_reason": f"Classification confidence too low ({confidence:.0%}). Audio may be too noisy or ambiguous."}

    quality = "good" if confidence >= 0.30 else "poor"
    quality_reason = sq_reason if quality == "good" else f"Low classification confidence ({confidence:.0%})"

    probe   = _pad_or_trim(y, CYCLE_SAMPLES)
    base_c  = float(_infer(_audio_to_tensor(probe))[top_idx])

    # Fine-grained occlusion sweep (0.5s windows) for evidence heatmap
    OCCL_WINDOW = 0.5  # seconds
    occl_samples = int(OCCL_WINDOW * SR)
    n_windows = int(CYCLE_SECS / OCCL_WINDOW)

    heatmap_timeline = []
    best_drop, salient = -1.0, (0.0, OCCL_WINDOW)
    for k in range(n_windows):
        t0_w = k * OCCL_WINDOW
        t1_w = t0_w + OCCL_WINDOW
        s0_w = int(t0_w * SR)
        s1_w = int(t1_w * SR)
        muted = probe.copy()
        muted[s0_w:s1_w] = 0.0
        drop = base_c - float(_infer(_audio_to_tensor(muted))[top_idx])
        heatmap_timeline.append((t0_w, t1_w, float(drop)))
        if drop > best_drop:
            best_drop = drop
            salient = (t0_w, t1_w)

    # Evidence regions: windows where drop is above 50% of peak drop
    drop_threshold = max(best_drop * 0.5, 0.01) if best_drop > 0 else 0.01
    evidence_regions = [(t0, t1, d) for t0, t1, d in heatmap_timeline if d >= drop_threshold]

    mel  = librosa.feature.melspectrogram(y=probe, sr=SR, n_mels=N_MELS,
                                           n_fft=N_FFT, hop_length=HOP_LENGTH)
    spec = librosa.power_to_db(mel, ref=np.max)
    gradcam_map = generate_gradcam(probe, target_label=label)

    return {
        "label":            label,
        "confidence":       confidence,
        "all_probs":        {c: float(p) for c, p in zip(CLASSES, mean_probs)},
        "quality":          quality,
        "quality_reason":   quality_reason,
        "cycles":           cycles_out,
        "salient":          salient,
        "heatmap_timeline": heatmap_timeline,
        "evidence_regions": evidence_regions,
        "spec":             spec,
        "gradcam":          gradcam_map,
    }
