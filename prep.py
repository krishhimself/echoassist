"""
prep.py  —  ICBHI 2017 -> mel spectrogram PNGs
Usage:
  .venv\\Scripts\\python.exe prep.py --test    (10 files only)
  .venv\\Scripts\\python.exe prep.py           (all 920 files)
"""

import sys
import random
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg

# ── constants ────────────────────────────────────────────────────────────────
SR            = 4000
CYCLE_SECS    = 5.0
CYCLE_SAMPLES = int(SR * CYCLE_SECS)   # 20 000 samples
MIN_SECS      = 0.5
N_MELS        = 64
SEED          = 42

DATA_DIR = Path("data/ICBHI_final_database")
OUT_DIR  = Path("spectrograms")

LABEL_MAP = {
    (0, 0): "normal",
    (1, 0): "crackle",
    (0, 1): "wheeze",
    (1, 1): "both",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def patient_id(wav_name: str) -> str:
    return wav_name.split("_")[0]


def pad_or_trim(audio: np.ndarray, target: int) -> np.ndarray:
    if len(audio) >= target:
        return audio[:target]
    return np.concatenate([audio, np.zeros(target - len(audio), dtype=audio.dtype)])


def process_file(wav_path: Path, split: str) -> dict[str, int]:
    txt_path = wav_path.with_suffix(".txt")
    if not txt_path.exists():
        print(f"  WARNING: no annotation for {wav_path.name} — skipped")
        return {}

    # Load at target SR, normalize
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    counts: dict[str, int] = {}

    with open(txt_path) as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        t_start, t_end   = float(parts[0]), float(parts[1])
        crackle, wheeze  = int(parts[2]), int(parts[3])

        if (t_end - t_start) < MIN_SECS:
            continue

        s0 = int(t_start * SR)
        s1 = int(t_end   * SR)
        cycle = pad_or_trim(y[s0:s1], CYCLE_SAMPLES)

        mel    = librosa.feature.melspectrogram(y=cycle, sr=SR, n_mels=N_MELS,
                                                n_fft=512, hop_length=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        cls     = LABEL_MAP[(crackle, wheeze)]
        out_dir = OUT_DIR / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{wav_path.stem}_{idx:03d}.png"
        # Flip vertically so low frequencies are at the bottom (visual convention)
        mpimg.imsave(out_dir / fname, np.flipud(mel_db), cmap="magma")

        counts[cls] = counts.get(cls, 0) + 1

    return counts


# ── main ─────────────────────────────────────────────────────────────────────

def main(test_mode: bool) -> None:
    wav_files = sorted(DATA_DIR.glob("*.wav"))
    if test_mode:
        wav_files = wav_files[:10]
        print(f"TEST MODE — processing {len(wav_files)} files\n")
    else:
        print(f"Processing all {len(wav_files)} files\n")

    # ── patient-level 80/20 split ────────────────────────────────────────────
    pid_to_files: dict[str, list[Path]] = {}
    for w in wav_files:
        pid = patient_id(w.name)
        pid_to_files.setdefault(pid, []).append(w)

    all_pids = sorted(pid_to_files)
    rng = random.Random(SEED)
    shuffled = all_pids.copy()
    rng.shuffle(shuffled)

    cut        = int(len(shuffled) * 0.8)
    train_pids = set(shuffled[:cut])
    test_pids  = set(shuffled[cut:])

    assert train_pids.isdisjoint(test_pids), "BUG: patient appears in both splits"

    print(f"Patients  — train: {len(train_pids)}, test: {len(test_pids)}")
    print()

    # ── process files ────────────────────────────────────────────────────────
    train_counts: dict[str, int] = {}
    test_counts:  dict[str, int] = {}

    for wav_path in wav_files:
        pid   = patient_id(wav_path.name)
        split = "train" if pid in train_pids else "test"
        counts = process_file(wav_path, split)
        target = train_counts if split == "train" else test_counts
        for cls, n in counts.items():
            target[cls] = target.get(cls, 0) + n

    # ── report ───────────────────────────────────────────────────────────────
    all_classes = sorted({"normal", "crackle", "wheeze", "both"})
    print("\n=== Cycles per class ===")
    print(f"{'class':<10}  {'train':>7}  {'test':>7}  {'total':>7}")
    print("-" * 38)
    for cls in all_classes:
        tr = train_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        print(f"{cls:<10}  {tr:>7}  {te:>7}  {tr+te:>7}")


if __name__ == "__main__":
    main(test_mode="--test" in sys.argv)
