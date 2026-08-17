"""
diag.py  --  compare predict.py preprocessing against the training pipeline
Loads 100 test-set breath cycles, runs predict_cycle() on each, and reports
accuracy + per-class recall.  Also cross-checks intermediate array shapes and
value ranges at each preprocessing step.
"""

import random
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch

# ── replicate the exact train/test patient split from prep.py ─────────────────
DATA_DIR  = Path("data/ICBHI_final_database")
SR        = 4000
SEED      = 42
LABEL_MAP = {(0,0):"normal",(1,0):"crackle",(0,1):"wheeze",(1,1):"both"}
CLASSES   = ["both","crackle","normal","wheeze"]   # alphabetical = ImageFolder order

def get_test_patients():
    wav_files = sorted(DATA_DIR.glob("*.wav"))
    all_pids  = sorted({w.name.split("_")[0] for w in wav_files})
    rng       = random.Random(SEED)
    shuffled  = all_pids.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return set(shuffled[cut:])

def parse_txt(txt_path):
    cycles = []
    for line in txt_path.read_text().splitlines():
        p = line.split()
        if len(p) >= 4 and p[0].replace(".","",1).isdigit():
            cycles.append((float(p[0]), float(p[1]), int(p[2]), int(p[3])))
    return cycles

# ── collect up to 100 test-set cycles (balanced across classes) ───────────────
def collect_cycles(n_total=100):
    test_pids = get_test_patients()
    per_class = defaultdict(list)

    for wav in sorted(DATA_DIR.glob("*.wav")):
        pid = wav.name.split("_")[0]
        if pid not in test_pids:
            continue
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        y, _ = librosa.load(str(wav), sr=SR)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        for t0, t1, cr, wh in parse_txt(txt):
            if (t1 - t0) < 0.5:
                continue
            label = LABEL_MAP[(cr, wh)]
            s0, s1 = int(t0*SR), int(t1*SR)
            per_class[label].append((y[s0:s1], label))
        # stop early once we have enough candidates
        if sum(len(v) for v in per_class.values()) >= n_total * 4:
            break

    # sample evenly; fall back to whatever exists for rare classes
    target_per = n_total // len(CLASSES)
    cycles = []
    for cls in CLASSES:
        items = per_class[cls]
        rng   = random.Random(SEED)
        rng.shuffle(items)
        cycles.extend(items[:target_per])
    random.Random(SEED).shuffle(cycles)
    return cycles[:n_total]

# ── preprocessing path A: predict.py's _audio_to_tensor ──────────────────────
# (import the real function so we test the actual code, not a copy)
from predict import _audio_to_tensor, _pad_or_trim, _infer, CYCLE_SAMPLES, N_MELS, N_FFT, HOP_LENGTH

def predict_path(audio_slice):
    """Exactly what predict_cycle() does internally."""
    cycle = _pad_or_trim(audio_slice, CYCLE_SAMPLES)
    tensor = _audio_to_tensor(cycle)
    probs  = _infer(tensor)
    return CLASSES[int(np.argmax(probs))], probs

# ── preprocessing path B: recreate what prep.py wrote to disk ─────────────────
# Replicates: melspectrogram -> power_to_db -> flipud -> imsave(cmap='magma')
# then loads it back as ImageFolder would (RGBA->RGB, ToTensor, Normalize)
from torchvision import transforms
_TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
from PIL import Image

def trainpath_tensor(audio_slice):
    """Reproduces what prep.py saved as a PNG, then loads it as ImageFolder would."""
    cycle  = _pad_or_trim(audio_slice, CYCLE_SAMPLES)
    mel    = librosa.feature.melspectrogram(y=cycle, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = np.flipud(mel_db)
    # matplotlib imsave normalises [vmin,vmax]->[0,1] then applies colormap
    # We'll literally call imsave to a temp buffer and reload, eliminating
    # any doubt about whether we're replicating it correctly.
    import io
    buf = io.BytesIO()
    plt.imsave(buf, img, cmap="magma", format="png")
    buf.seek(0)
    pil_rgba = Image.open(buf).convert("RGBA")
    pil_rgb  = pil_rgba.convert("RGB")
    return _TRANSFORM_TRAIN(pil_rgb).unsqueeze(0)

def trainpath_predict(audio_slice):
    tensor = trainpath_tensor(audio_slice)
    with torch.no_grad():
        from predict import _load_model
        probs = torch.softmax(_load_model()(tensor), dim=1)[0].numpy()
    return CLASSES[int(np.argmax(probs))], probs

# ── shape / value sanity checks ───────────────────────────────────────────────
def check_preprocessing():
    print("=== Preprocessing sanity checks ===")
    # use a deterministic test signal: 5s of 200Hz sine
    t     = np.linspace(0, 5, CYCLE_SAMPLES, endpoint=False)
    dummy = np.sin(2 * np.pi * 200 * t).astype(np.float32)

    cycle = _pad_or_trim(dummy, CYCLE_SAMPLES)
    print(f"pad_or_trim output: {len(cycle)} samples  (expected {CYCLE_SAMPLES})")

    # check a short clip gets padded
    short = dummy[:3000]
    padded = _pad_or_trim(short, CYCLE_SAMPLES)
    assert len(padded) == CYCLE_SAMPLES, "pad_or_trim failed for short clip"
    assert np.all(padded[3000:] == 0.0), "padding is not zeros"
    print(f"Short clip ({len(short)} samples) padded to {len(padded)}  OK")

    mel    = librosa.feature.melspectrogram(y=cycle, sr=SR, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    print(f"mel shape: {mel_db.shape}  (expected ({N_MELS}, ~157))")
    print(f"mel_db range: [{mel_db.min():.1f}, {mel_db.max():.1f}] dB")

    tensor_pred  = _audio_to_tensor(cycle)             # predict.py path
    tensor_train = trainpath_tensor(cycle)              # round-trip via imsave
    print(f"predict.py tensor shape:   {tuple(tensor_pred.shape)}")
    print(f"train-path tensor shape:   {tuple(tensor_train.shape)}")

    diff = (tensor_pred - tensor_train).abs()
    print(f"Max pixel diff between paths: {diff.max().item():.4f}  "
          f"(mean: {diff.mean().item():.4f})")
    if diff.max().item() < 0.05:
        print("  -> paths match closely  OK")
    else:
        print("  -> MISMATCH WARNING -- preprocessing diverges!")
    print()

# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(cycles):
    correct_pred  = defaultdict(int)
    correct_train = defaultdict(int)
    total         = defaultdict(int)

    for audio, gt in cycles:
        total[gt] += 1

        pred_lbl,  _ = predict_path(audio)
        train_lbl, _ = trainpath_predict(audio)

        if pred_lbl  == gt: correct_pred[gt]  += 1
        if train_lbl == gt: correct_train[gt] += 1

    print("=== Per-class results (100 test cycles) ===")
    print(f"{'class':<10} {'GT count':>9} | {'predict.py':>10} {'recall':>7} | {'train path':>10} {'recall':>7}")
    print("-" * 65)

    tot_pred = tot_train = tot_all = 0
    for cls in CLASSES:
        n  = total[cls]
        cp = correct_pred[cls]
        ct = correct_train[cls]
        tot_all   += n
        tot_pred  += cp
        tot_train += ct
        r_pred  = cp/n if n else 0
        r_train = ct/n if n else 0
        print(f"{cls:<10} {n:>9} | {cp:>10} {r_pred:>7.3f} | {ct:>10} {r_train:>7.3f}")

    print("-" * 65)
    acc_pred  = tot_pred  / tot_all
    acc_train = tot_train / tot_all
    print(f"{'OVERALL':<10} {tot_all:>9} | {tot_pred:>10} {acc_pred:>7.3f} | {tot_train:>10} {acc_train:>7.3f}")

    print()
    if abs(acc_pred - acc_train) < 0.05:
        print("predict.py and train-path agree  -> no preprocessing divergence")
    else:
        print("WARNING: predict.py and train-path disagree by "
              f"{abs(acc_pred-acc_train):.1%}  -> investigate _audio_to_tensor")

if __name__ == "__main__":
    check_preprocessing()

    print("Collecting 100 test-set cycles...")
    cycles = collect_cycles(100)
    count  = {cls: sum(1 for _,l in cycles if l==cls) for cls in CLASSES}
    print(f"Collected: {dict(count)}  total={len(cycles)}\n")

    evaluate(cycles)
