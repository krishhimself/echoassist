"""
app.py — Streamlit dashboard for EchoAssist

Run:
  .venv\\Scripts\\streamlit.exe run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from predict import analyze, CLASSES, SR, HOP_LENGTH

# ── palette (validated categorical order — see dataviz skill) ──────────────
COLORS = {
    "normal":  "#2a78d6",  # slot 1 blue
    "crackle": "#eb6834",  # slot 2 orange
    "wheeze":  "#1baf7a",  # slot 3 aqua
    "both":    "#eda100",  # slot 4 yellow
}
SURFACE     = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED   = "#898781"
GRIDLINE    = "#e1e0d9"
BASELINE    = "#c3c2b7"

st.set_page_config(page_title="EchoAssist", layout="centered")

st.title("EchoAssist")
st.caption("Describes lung sounds from audio. Not a diagnosis.")

# ── input ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a respiratory recording (.wav)", type=["wav"])

sample_dir = Path("data/ICBHI_final_database")
sample_files = sorted(sample_dir.glob("*.wav")) if sample_dir.exists() else []
sample_choice = None
if sample_files:
    sample_choice = st.selectbox(
        "…or pick a sample file", ["(none)"] + [p.name for p in sample_files]
    )

audio_source = None
if uploaded is not None:
    audio_source = uploaded
elif sample_choice and sample_choice != "(none)":
    audio_source = sample_dir / sample_choice

if audio_source is None:
    st.info("Upload a .wav file or pick a sample to analyze.")
    st.stop()

st.audio(audio_source)

# ── run analysis ─────────────────────────────────────────────────────────
result = analyze(audio_source)

if result["quality"] == "poor":
    st.error(result["message"] or "signal quality too low, re-record")
    st.stop()

# ── headline ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("Predicted label", result["label"])
col2.metric("Confidence", f"{result['confidence']:.0%}")

# ── bar chart: class probabilities ──────────────────────────────────────
st.subheader("Class probabilities")

fig, ax = plt.subplots(figsize=(6, 3))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

probs = result["all_probs"]
labels = CLASSES
values = [probs[c] for c in labels]
bar_colors = [COLORS[c] for c in labels]

bars = ax.bar(labels, values, color=bar_colors, width=0.6)
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
        ha="center", va="bottom", color=INK_PRIMARY, fontsize=10,
    )

ax.set_ylim(0, 1.0)
ax.set_ylabel("probability", color=INK_MUTED)
ax.tick_params(colors=INK_MUTED)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
ax.set_axisbelow(True)

st.pyplot(fig)
plt.close(fig)

# ── spectrogram ──────────────────────────────────────────────────────────
st.subheader("Mel spectrogram")

spec = result["spec"]
duration = spec.shape[1] * HOP_LENGTH / SR

fig2, ax2 = plt.subplots(figsize=(8, 3))
fig2.patch.set_facecolor(SURFACE)
im = ax2.imshow(
    spec, cmap="magma", aspect="auto", origin="upper",
    extent=[0, duration, 0, spec.shape[0]],
)
ax2.set_xlabel("time (s)", color=INK_MUTED)
ax2.set_ylabel("mel bin", color=INK_MUTED)
ax2.tick_params(colors=INK_MUTED)
for spine in ax2.spines.values():
    spine.set_color(BASELINE)

# highlight the salient region the model relied on most
if result["salient"] is not None:
    s0, s1 = result["salient"]
    ax2.axvspan(s0, s1, color="white", alpha=0.15, edgecolor="white", linewidth=1.5)

st.pyplot(fig2)
plt.close(fig2)

if result["salient"] is not None:
    s0, s1 = result["salient"]
    st.caption(f"Most influential region: {s0:.1f}s – {s1:.1f}s (shaded above)")

# ── cycle timeline ───────────────────────────────────────────────────────
st.subheader("Breath cycle timeline")

fig3, ax3 = plt.subplots(figsize=(8, 1.2))
fig3.patch.set_facecolor(SURFACE)
ax3.set_facecolor(SURFACE)

for start, end, cyc_label in result["cycles"]:
    ax3.broken_barh(
        [(start, end - start)], (0, 1),
        facecolors=COLORS.get(cyc_label, INK_MUTED),
        edgecolors=SURFACE, linewidth=2,
    )
    ax3.text(
        (start + end) / 2, 0.5, cyc_label,
        ha="center", va="center", color="white", fontsize=8, fontweight="bold",
    )

ax3.set_xlim(0, duration)
ax3.set_ylim(0, 1)
ax3.set_yticks([])
ax3.set_xlabel("time (s)", color=INK_MUTED)
ax3.tick_params(colors=INK_MUTED)
for spine in ax3.spines.values():
    spine.set_visible(False)

st.pyplot(fig3)
plt.close(fig3)

# legend for the timeline (categorical identity)
legend_cols = st.columns(len(CLASSES))
for col, cls in zip(legend_cols, CLASSES):
    col.markdown(
        f'<span style="color:{COLORS[cls]}">■</span> {cls}',
        unsafe_allow_html=True,
    )
