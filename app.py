"""
app.py  —  EchoAssist Streamlit dashboard
Run: .venv\\Scripts\\streamlit.exe run app.py
"""

import tempfile
from pathlib import Path

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from predict import (
    CLASSES, HOP_LENGTH, N_FFT, N_MELS, SR,
    analyze, predict_cycle,
)

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/ICBHI_final_database")
SAMPLES_DIR = Path("samples")

LABEL_COLORS = {
    "normal":  "#4CAF50",
    "crackle": "#FF9800",
    "wheeze":  "#2196F3",
    "both":    "#E91E63",
}
LABEL_MAP = {(0, 0): "normal", (1, 0): "crackle", (0, 1): "wheeze", (1, 1): "both"}

# Curated sample list: (friendly label, filename, has_annotation)
# Files without annotation (synthetic) go through analyze() graceful-refusal path.
SAMPLES = [
    ("Normal - patient 210 (left chest)",  "normal_patient210_Al.wav",  True),
    ("Normal - patient 112 (right chest)", "normal_patient112_Ar.wav",  True),
    ("Crackle - patient 223 (lower right)","crackle_patient223_Lr.wav", True),
    ("Crackle - patient 205 (right)",      "crackle_patient205_Ar.wav", True),
    ("Wheeze - patient 223 (right)",       "wheeze_patient223_Ar.wav",  True),
    ("Wheeze - patient 206 (lower left)",  "wheeze_patient206_Pl.wav",  True),
    ("Both - patient 156 (right)",         "both_patient156_Pr.wav",    True),
    ("Silent file (bad input test)",       "silent.wav",                False),
    ("Too-short file (bad input test)",    "short.wav",                 False),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_annotation(txt_path: Path) -> list:
    """Parse ICBHI .txt into [(t0, t1, label), ...]. Skips non-numeric header lines."""
    cycles = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].replace(".", "", 1).isdigit():
                t0, t1 = float(parts[0]), float(parts[1])
                label  = LABEL_MAP[(int(parts[2]), int(parts[3]))]
                cycles.append((t0, t1, label))
    return cycles


@st.cache_data
def gallery_files() -> list:
    """Return the curated samples list as (label, wav_path, txt_path_or_None)."""
    entries = []
    for label, fname, has_ann in SAMPLES:
        wav = SAMPLES_DIR / fname
        if not wav.exists():
            continue
        # look for matching annotation in data/ by patient ID embedded in filename
        txt = None
        if has_ann:
            # filename pattern: {class}_patient{pid}_{loc}.wav
            # original stem stored next to the wav file as a .txt with same name
            stem_txt = SAMPLES_DIR / fname.replace(".wav", ".txt")
            if stem_txt.exists():
                txt = stem_txt
            else:
                # fall back to searching data/ for the original file
                # filename encodes original: e.g. normal_patient210_Al.wav
                # -> original is 210_*_Al_sc_Meditron.wav
                parts = fname.replace(".wav", "").split("_")
                if len(parts) >= 3:
                    pid = parts[1].replace("patient", "")
                    loc = parts[2]
                    matches = list(DATA_DIR.glob(f"{pid}_*_{loc}_*.txt"))
                    if matches:
                        txt = matches[0]
        entries.append((label, wav, txt))
    return entries


@st.cache_data
def load_audio(wav_path: str) -> np.ndarray:
    """Load and normalise a wav file at SR=4000. Cached by path string."""
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def run_gallery_inference(y: np.ndarray, gt_cycles: list) -> list:
    """
    For each annotated cycle: slice audio, run model.
    Returns [(t0, t1, pred_label, gt_label), ...].
    """
    results = []
    for (t0, t1, gt_label) in gt_cycles:
        if (t1 - t0) < 0.5:
            continue
        s0, s1 = int(t0 * SR), int(t1 * SR)
        pred_label, _, _ = predict_cycle(y[s0:s1])
        results.append((t0, t1, pred_label, gt_label))
    return results


def full_spectrogram(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_spectrogram(spec: np.ndarray, duration: float,
                     salient: tuple | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.imshow(np.flipud(spec), aspect="auto", origin="upper", cmap="magma",
              extent=[0, duration, 0, N_MELS])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title("Mel spectrogram")
    if salient:
        ax.axvspan(salient[0], salient[1], color="cyan", alpha=0.25, label="salient")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_timeline(rows: list[tuple], row_labels: list[str], duration: float) -> plt.Figure:
    """
    rows        — list of lists; each list is [(t0, t1, label), ...]
    row_labels  — y-axis tick labels, one per row
    """
    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(10, 0.8 * n_rows + 0.6))

    for row_idx, (cycle_list, row_label) in enumerate(zip(rows, row_labels)):
        for (t0, t1, label) in cycle_list:
            width = t1 - t0
            ax.barh(row_idx, width, left=t0, height=0.55,
                    color=LABEL_COLORS.get(label, "#999"),
                    align="center", edgecolor="white", linewidth=0.4)
            if width > 0.8:
                ax.text((t0 + t1) / 2, row_idx, label,
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.grid(axis="x", alpha=0.25)

    patches = [mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()]
    ax.legend(handles=patches, ncol=4, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, 1.18))

    fig.tight_layout()
    return fig


def plot_confidence(probs: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 2.2))
    labels  = list(probs.keys())
    values  = [probs[l] for l in labels]
    colors  = [LABEL_COLORS[l] for l in labels]
    bars    = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Class probabilities")
    for bar, v in zip(bars, values):
        ax.text(min(v + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ── page layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="EchoAssist", layout="wide")
st.title("EchoAssist - Lung Sound Classifier")
st.caption("Classifies respiratory audio into: normal / crackle / wheeze / both")

# ── sidebar: both input controls always visible ───────────────────────────────
with st.sidebar:
    st.header("Audio input")

    st.subheader("Sample gallery")
    files        = gallery_files()
    label_options = [lbl for lbl, _, _ in files]
    choice       = st.selectbox("Sample", label_options, label_visibility="collapsed")

    st.divider()

    st.subheader("Upload your own file")
    uploaded = st.file_uploader("WAV / MP3 / FLAC / OGG",
                                type=["wav", "mp3", "flac", "ogg"],
                                label_visibility="collapsed")
    if uploaded:
        st.caption("Uploaded file takes priority over gallery selection.")

# ════════════════════════════════════════════════════════════════════════════
# UPLOAD MODE  —  analyze() with fixed 5s windows + salient region
# (shown when a file is uploaded; gallery is ignored)
# ════════════════════════════════════════════════════════════════════════════
def show_analyze_result(audio_source, display_name: str) -> None:
    """Run analyze() and render results. audio_source: file path or file-like."""
    import pandas as pd

    if hasattr(audio_source, "read"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_source.read())
            tmp_path = tmp.name
        result = analyze(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
    else:
        result = analyze(str(audio_source))

    if "error" in result:
        st.error(result["error"])
        return

    quality_tag = "[good]" if result["quality"] == "good" else "[poor quality]"
    st.markdown(
        f"### Prediction: **{result['label'].upper()}**  "
        f"-- confidence {result['confidence']:.0%}  {quality_tag}"
    )
    st.caption("This describes the sound only, not a medical diagnosis.")

    col_spec, col_conf = st.columns([3, 1])
    with col_spec:
        st.subheader("Spectrogram (first 5 s)")
        st.pyplot(
            plot_spectrogram(result["spec"], 5.0, salient=result["salient"]),
            use_container_width=True,
        )
        st.caption(
            f"Cyan band = most influential 1-second region "
            f"({result['salient'][0]:.0f}--{result['salient'][1]:.0f} s)"
        )
    with col_conf:
        st.subheader("Probabilities")
        st.pyplot(plot_confidence(result["all_probs"]), use_container_width=True)

    if result["cycles"]:
        st.subheader("Predicted cycles")
        cycles    = result["cycles"]
        total_dur = cycles[-1][1]
        st.pyplot(
            plot_timeline(
                rows=[[(t0, t1, lbl) for t0, t1, lbl in cycles]],
                row_labels=["Predicted"],
                duration=total_dur,
            ),
            use_container_width=True,
        )


if uploaded:
    st.subheader(f"Uploaded: {uploaded.name}")
    st.audio(uploaded)
    with st.spinner("Analysing uploaded file…"):
        show_analyze_result(uploaded, uploaded.name)

# ════════════════════════════════════════════════════════════════════════════
# GALLERY MODE  —  annotated files get two-row timeline (predicted + GT);
#                  synthetic bad-input files go through analyze() graceful refusal
# ════════════════════════════════════════════════════════════════════════════
else:
    import pandas as pd

    _, wav_path, txt_path = next(f for f in files if f[0] == choice)

    st.subheader(choice)
    st.audio(str(wav_path))

    # ── synthetic / bad-input samples: no annotation, route through analyze() ──
    if txt_path is None:
        with st.spinner("Analysing…"):
            show_analyze_result(wav_path, choice)

    # ── annotated ICBHI sample: real cycle boundaries, two-row timeline ─────────
    else:
        y         = load_audio(str(wav_path))
        duration  = len(y) / SR
        gt_cycles = load_annotation(txt_path)

        st.caption(f"Duration: {duration:.1f} s  |  Annotated cycles: {len(gt_cycles)}")

        with st.spinner("Running inference on each annotated cycle…"):
            results = run_gallery_inference(y, gt_cycles)

        if not results:
            st.warning("No usable cycles found in annotation.")
        else:
            correct = sum(pred == gt for _, _, pred, gt in results)
            st.metric("Cycle accuracy on this recording",
                      f"{correct}/{len(results)} = {correct/len(results):.0%}")

            disp_y    = y[:int(min(duration, 60) * SR)]
            disp_dur  = len(disp_y) / SR
            disp_spec = full_spectrogram(disp_y)

            st.subheader("Spectrogram")
            st.pyplot(plot_spectrogram(disp_spec, disp_dur), use_container_width=True)

            st.subheader("Cycle timeline")
            pred_row = [(t0, t1, pred) for t0, t1, pred, _ in results]
            gt_row   = [(t0, t1, gt)   for t0, t1, _,    gt in results]
            st.pyplot(
                plot_timeline(
                    rows=[pred_row, gt_row],
                    row_labels=["Predicted", "Ground Truth"],
                    duration=disp_dur,
                ),
                use_container_width=True,
            )

            with st.expander("Per-cycle detail"):
                df = pd.DataFrame(
                    [(f"{t0:.2f}", f"{t1:.2f}", pred, gt, "Y" if pred == gt else "N")
                     for t0, t1, pred, gt in results],
                    columns=["Start (s)", "End (s)", "Predicted", "Ground Truth", "Match"],
                )
                st.dataframe(df, use_container_width=True)
