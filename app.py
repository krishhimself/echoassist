"""
app.py  —  EchoAssist Streamlit dashboard (Multi-Label Classification + Grad-CAM Integration)
Run: .venv\Scripts\streamlit.exe run app.py
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from predict import (
    CLASSES, HOP_LENGTH, N_FFT, N_MELS, SR,
    analyze, predict_cycle, generate_gradcam,
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

def create_salient_audio_bytes(audio_source, salient: tuple[float, float]) -> bytes | None:
    """Extract audio slice corresponding to salient (t0, t1) seconds and return WAV bytes."""
    try:
        if hasattr(audio_source, "read"):
            audio_source.seek(0)
            y, _ = librosa.load(audio_source, sr=SR)
        else:
            y, _ = librosa.load(str(audio_source), sr=SR)
        t0, t1 = salient
        s0, s1 = int(t0 * SR), int(t1 * SR)
        slice_y = y[s0:s1]
        if len(slice_y) == 0:
            return None
        buf = io.BytesIO()
        sf.write(buf, slice_y, SR, format="WAV")
        return buf.getvalue()
    except Exception:
        return None


def load_annotation(txt_path: Path) -> list:
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
    entries = []
    for label, fname, has_ann in SAMPLES:
        wav = SAMPLES_DIR / fname
        if not wav.exists():
            continue
        txt = None
        if has_ann:
            stem_txt = SAMPLES_DIR / fname.replace(".wav", ".txt")
            if stem_txt.exists():
                txt = stem_txt
            else:
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
    y, _ = librosa.load(wav_path, sr=SR)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def run_gallery_inference(y: np.ndarray, gt_cycles: list) -> list:
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
                     salient: tuple | None = None,
                     gradcam: np.ndarray | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.imshow(np.flipud(spec), aspect="auto", origin="upper", cmap="magma",
              extent=[0, duration, 0, N_MELS])
    if gradcam is not None:
        ax.imshow(gradcam, aspect="auto", origin="upper", cmap="jet",
                  alpha=0.45, extent=[0, duration, 0, N_MELS])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    title = "Mel Spectrogram + Grad-CAM Heatmap" if gradcam is not None else "Mel Spectrogram"
    ax.set_title(title)
    if salient:
        ax.axvspan(salient[0], salient[1], color="cyan", alpha=0.3, label="Salient region")
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_timeline(rows: list[tuple], row_labels: list[str], duration: float) -> plt.Figure:
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
st.title("EchoAssist - Multi-Label Lung Sound Classifier & Grad-CAM Explainability")
st.caption("Multi-label binary classification: normal / crackle / wheeze / both with Grad-CAM feature attribution")

# ── sidebar: input controls & model benchmark ────────────────────────────────
with st.sidebar:
    st.header("Audio input")

    st.subheader("Sample gallery")
    files        = gallery_files()
    label_options = [lbl for lbl, _, _ in files]
    choice       = st.selectbox("Sample", label_options, label_visibility="collapsed")

    st.divider()

    st.subheader("Upload audio file")
    uploaded = st.file_uploader("WAV / MP3 / FLAC / OGG",
                                type=["wav", "mp3", "flac", "ogg"],
                                label_visibility="collapsed")

    st.divider()

    st.subheader("Record from microphone")
    recorded = st.audio_input("Record audio", label_visibility="collapsed")

    if uploaded:
        st.caption("ℹ️ Uploaded file takes priority.")
    elif recorded:
        st.caption("ℹ️ Live microphone recording takes priority.")

    st.divider()

    with st.expander("📊 Model Benchmark & Architecture"):
        st.markdown(
            "**Multi-Label Model Upgrade**\n"
            "- **Architecture:** ResNet18 Multi-Label Binary Classification\n"
            "- **Explainability:** Grad-CAM Layer4 Heatmaps & Occlusion Analysis\n"
            "- **Evaluation Split:** Patient-Independent (100 train / 26 test)\n"
            "- **Macro Recall:** `40.0%` (Baseline: `25.0%`)"
        )
        if Path("confusion_matrix.png").exists():
            st.image("confusion_matrix.png", caption="Patient-Independent Confusion Matrix")


# ════════════════════════════════════════════════════════════════════════════
# ANALYZE DISPLAY  —  analyze() + Grad-CAM Heatmap + Occlusion + Report
# ════════════════════════════════════════════════════════════════════════════
def show_analyze_result(audio_source, display_name: str) -> None:
    import pandas as pd

    if hasattr(audio_source, "read"):
        audio_source.seek(0)
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
        tab1, tab2 = st.tabs(["📊 Mel Spectrogram & Salient Audio", "🔥 Grad-CAM Feature Heatmap"])

        with tab1:
            st.pyplot(
                plot_spectrogram(result["spec"], 5.0, salient=result["salient"]),
                use_container_width=True,
            )
            st.caption(
                f"Cyan band = most influential 1-second region "
                f"({result['salient'][0]:.0f}--{result['salient'][1]:.0f} s)"
            )
            salient_bytes = create_salient_audio_bytes(audio_source, result["salient"])
            if salient_bytes:
                st.audio(salient_bytes, format="audio/wav")
                st.caption(
                    f"🔊 **Play Salient 1-Second Audio** "
                    f"({result['salient'][0]:.0f}--{result['salient'][1]:.0f} s) — What the AI heard"
                )

        with tab2:
            if "gradcam" in result:
                st.pyplot(
                    plot_spectrogram(result["spec"], 5.0, gradcam=result["gradcam"]),
                    use_container_width=True,
                )
                st.caption(
                    "🔥 **Grad-CAM Heatmap:** Red/Yellow regions highlight exact spatial points "
                    "in the Mel-Spectrogram that drove ResNet18 convolutional neural network activations."
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

    st.divider()
    report_text = f"""==================================================
ECHOASSIST CLINICAL ACOUSTIC SUMMARY REPORT
==================================================
Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source File:  {display_name}

ACOUSTIC CLASSIFICATION RESULTS:
--------------------------------------------------
Primary Sound Feature: {result['label'].upper()}
Model Confidence:      {result['confidence']:.1%}
Signal Quality Status: {result['quality'].upper()}
Salient Audio Window:  {result['salient'][0]:.1f}s - {result['salient'][1]:.1f}s

CLASS PROBABILITY DISTRIBUTION:
--------------------------------------------------
"""
    for cls_name, prob_val in result["all_probs"].items():
        report_text += f"  - {cls_name.capitalize():<10}: {prob_val:.1%}\n"

    report_text += """
==================================================
DISCLAIMER:
EchoAssist describes acoustic sound features only. 
This software is intended for clinical decision 
support and does NOT output a medical diagnosis.
==================================================
"""
    st.download_button(
        label="📄 Download Summary Report (.txt)",
        data=report_text,
        file_name=f"echoassist_report_{result['label']}.txt",
        mime="text/plain",
    )


# ════════════════════════════════════════════════════════════════════════════
# ROUTING  — Uploaded File -> Live Mic -> Sample Gallery
# ════════════════════════════════════════════════════════════════════════════
if uploaded:
    st.subheader(f"Uploaded: {uploaded.name}")
    st.audio(uploaded)
    with st.spinner("Analysing uploaded file with Grad-CAM…"):
        show_analyze_result(uploaded, uploaded.name)

elif recorded:
    st.subheader("Microphone Recording")
    st.audio(recorded)
    with st.spinner("Analysing microphone recording with Grad-CAM…"):
        show_analyze_result(recorded, "Microphone_Recording.wav")

else:
    import pandas as pd

    _, wav_path, txt_path = next(f for f in files if f[0] == choice)

    st.subheader(choice)
    st.audio(str(wav_path))

    if txt_path is None:
        with st.spinner("Analysing…"):
            show_analyze_result(wav_path, choice)

    else:
        y         = load_audio(str(wav_path))
        duration  = len(y) / SR
        gt_cycles = load_annotation(txt_path)

        st.caption(f"Duration: {duration:.1f} s  |  Annotated cycles: {len(gt_cycles)}")

        with st.spinner("Running multi-label inference on each annotated cycle…"):
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
            gradcam_map = generate_gradcam(disp_y[:CYCLE_SAMPLES])

            tab1, tab2 = st.tabs(["📊 Mel Spectrogram", "🔥 Grad-CAM Feature Heatmap"])
            with tab1:
                st.pyplot(plot_spectrogram(disp_spec, disp_dur), use_container_width=True)
            with tab2:
                st.pyplot(plot_spectrogram(disp_spec, disp_dur, gradcam=gradcam_map), use_container_width=True)
                st.caption("🔥 **Grad-CAM Activation Heatmap:** Shows spatial frequency & time regions that activated CNN layers.")

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
