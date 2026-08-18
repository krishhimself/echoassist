# EchoAssist

**Acoustic analysis and clinical decision support for lung sounds.**

Classifies respiratory recordings as normal, crackle, wheeze, or both — with a confidence score, an explanation of which part of the audio drove the decision, and a deliberate refusal when the signal is too poor to analyse.

Built for PS-S01, Healthcare domain, 24-hour hackathon.

> EchoAssist describes acoustic patterns. It does not diagnose.

---

## The core idea

**Sound becomes a picture, then we do image recognition on the picture.**

A spectrogram is a picture of sound — time on the horizontal axis, frequency on the vertical, brightness as energy. Wheezes appear as long horizontal streaks; crackles as short vertical specks. Once audio is an image, the problem becomes image classification, which is the most solved problem in machine learning.

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 49.3% |
| Macro-average recall | 0.377 baseline, 0.400 with augmentation |
| Test set | 1,885 breath cycles from 26 unseen patients |
| Random baseline | 25% (4 classes) |
| Always-predict-normal baseline | 44% accuracy, 0.25 macro-recall |

Per-class recall: normal 0.611 · crackle 0.513 · wheeze 0.283 · both 0.103

**Why this number is honest.** Our train/test split is patient-independent — no patient appears in both sets. Reports of 85–95% on ICBHI typically use file-level splits, which leak the same patient into both and inflate results substantially. We also report macro-recall rather than accuracy, because always predicting "normal" would score 44% accuracy while being clinically worthless.

---

## Quick start

**Requirements:** Python 3.11+, ~2 GB disk for the dataset.

```bash
git clone https://github.com/krishhimself/echoassist
cd echoassist
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install librosa matplotlib numpy torch torchvision streamlit scikit-learn soundfile
pip install -r requirements-api.txt
```

### Run the Streamlit app

```bash
python -m streamlit run app.py
```

Opens at **http://localhost:8501**. This is the reference implementation — everything works here.

### Run the API + web frontend

Two terminals.

```bash
# Terminal 1 — backend
python -m uvicorn api:app --reload --port 8080
```

```bash
# Terminal 2 — frontend
cd frontend
python -m http.server 3000
```

Frontend at **http://localhost:3000**. Interactive API docs at **http://localhost:8080/docs**.

Start the backend first — the frontend shows "API offline" without it.

**Demo login:** `demo` / `demo123`

---

## Reproducing the model

Download the **ICBHI 2017 Respiratory Sound Database** (search Kaggle for "Respiratory Sound Database"), unzip into `data/`, then:

```bash
python prep.py     # 920 wav files -> 6,857 spectrogram PNGs, ~2 min
python train.py    # ResNet18 transfer learning -> model.pth, ~23 min on CPU
python metrics.py  # accuracy, per-class precision/recall/F1, confusion matrix
python diag.py     # verifies training and inference preprocessing are identical
```

No GPU required.

---

## Project structure

```
echoassist/
├── prep.py          Audio -> mel spectrogram PNGs, patient-level split
├── train.py         ResNet18 transfer learning
├── train2.py        Same, with SpecAugment and stronger regularisation
├── predict.py       analyze() — the single inference interface
├── diag.py          Preprocessing parity verification
├── metrics.py       Evaluation and confusion matrix
├── api.py           FastAPI wrapper around analyze()
├── auth.py          Clinician login (demo accounts)
├── db.py            SQLite — patients and analysis history
├── app.py           Streamlit dashboard
├── frontend/        Web frontend (HTML/CSS/JS)
├── samples/         Curated demo recordings
└── CLAUDE.md        Project spec and constraints
```

---

## How it works

```
Audio file
    ↓ load at 4000 Hz, peak normalise
Quality gate ──── fail → "No reliable signal detected"
    ↓ duration, clipping, SNR checks
Segmentation
    ↓ real breath-cycle boundaries, or 5s windows
Mel spectrogram
    ↓ n_mels=64, n_fft=512, hop_length=128  →  3×64×157
ResNet18 (transfer-learned, final layer replaced)
    ↓ softmax
Explainability
    ↓ occlusion analysis + Grad-CAM
Result: label, score, probabilities, salient region, stability
```

### Key parameters

```python
SR         = 4000    # Hz — Nyquist covers lung sounds below 2 kHz
N_MELS     = 64      # mel frequency bands
n_fft      = 512     # 128 ms analysis window
hop_length = 128     # 32 ms between frames
CYCLE_SECS = 5.0     # every segment padded/trimmed to this
```

These must be identical in `prep.py` and `predict.py`. A mismatch produces confident nonsense while every line of code appears to run correctly. `diag.py` verifies this: **pixel difference 0.0000** between the training and inference paths.

---

## Design decisions

**One breath cycle is one training example, not one file.** Each `.wav` ships with a `.txt` annotating every respiratory cycle and whether it contains crackles or wheezes. Using cycles rather than files turns 920 recordings into 6,857 labelled examples — a 7.5× increase from data already in the download.

**Split by patient, never by file.** Multiple recordings come from the same patient, on the same equipment, in the same room. If a patient appears in both train and test, the model learns to recognise their recording setup rather than their pathology. The split runs over patient IDs before any file is opened, followed by an assertion that the sets are disjoint. This costs roughly 20–30 percentage points of headline accuracy and is the single most important correctness property of the project.

**The spectrogram window was tuned for crackles.** librosa's default `n_fft=2048` gives a 512 ms analysis window at 4 kHz. A crackle lasts 5–20 ms — inside that window it contributes about 2% of the energy and is smeared into invisibility. The model would have been structurally incapable of detecting 27% of our data, while overall accuracy still looked acceptable. We use `n_fft=512` for a 128 ms window.

**Selection by macro-recall, not accuracy.** Accuracy is dominated by the majority class. Macro-recall weights every class equally, so it cannot be gamed by ignoring rare classes.

**Augmentation, not more epochs.** Train loss reached 0.09 while test macro-recall plateaued at 0.38 — the signature of memorisation. More epochs would have deepened it. SpecAugment kept train loss at 0.69 while macro-recall improved to 0.400 and the rarest class more than doubled, from 0.103 to 0.239.

**Refusal is a feature.** Recordings that are too short, too quiet, clipped, or too noisy are declined rather than guessed at. In a clinical context a confident wrong answer is more dangerous than no answer.

---

## Explainability

**Occlusion analysis.** Each one-second chunk is muted in turn and the model re-run. The chunk causing the largest drop in confidence is the region the model relied on. This is a direct causal measurement — if removing a second collapses confidence, the model was demonstrably using it. The interface then plays that exact second, so you can hear what the model heard.

**Grad-CAM.** A gradient-based attention map over the spectrogram. Two independent methods agreeing is stronger evidence than either alone.

---

## Limitations

- Rare classes are weak. `both` is 7% of the data and requires detecting two phenomena simultaneously.
- Softmax output is **not calibrated probability**. Neural networks are systematically overconfident, which is why the interface labels it a model score rather than a confidence.
- Trained on ICBHI only — performance on other stethoscopes, populations, or recording conditions is unverified.
- Demo authentication uses fixed accounts. The authorisation model is correct — every query is scoped by clinician — but production would need real identity management.
- **Not a medical device.** No clinical validation. Research and demonstration only.

---

## Dataset

Rocha et al., *An open access database for the evaluation of respiratory sound classification algorithms*, ICBHI 2017. 920 recordings, 126 patients, annotated at respiratory-cycle level.

---

## License

MIT
