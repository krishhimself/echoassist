# EchoAssist — lung sound classifier (24-hour hackathon)

## Goal
Classify respiratory audio into: normal / crackle / wheeze / both.
Deliverables required by the problem statement:
- Noise reduction before analysis
- Feature extraction (spectrograms, wheeze/crackle segmentation)
- Classifier: normal + 3 abnormal classes, with confidence scores
- Explainability: show WHICH part of the audio drove the decision
- Streamlit dashboard demoing the pipeline end-to-end
- Metrics: accuracy, precision, recall, F1 per class
- Graceful handling of silent / corrupt / unusable audio (never crash)
Scope limit: describe the SOUND only. Never output a diagnosis.

## Environment — confirmed, do not change or "upgrade"
- Windows, PowerShell
- Python 3.14, virtualenv at .venv
- librosa 1.0.0, torch 2.13.0+cpu, torchvision, streamlit, scikit-learn
- CPU ONLY. No GPU. torch.cuda.is_available() is False.
- ALWAYS run python as `.venv\Scripts\python.exe`, never bare `python`.
  Bare python misses the venv and fails with "no module named librosa".

## CPU constraints — these drive design choices
- Keep spectrograms small, ~64 px tall. Do NOT resize to 224x224.
- Prefer ResNet18 over anything larger.
- Avoid librosa.util.* helpers (1.0 API uncertain) — use numpy for
  padding and trimming.
- Mel spectrogram params (MUST be identical in prep.py and predict.py):
  n_mels=64, n_fft=512, hop_length=128, sr=4000
  (n_fft=512 → 128ms window, resolves crackles which are 5-20ms transients)

## Dataset — ICBHI 2017 Respiratory Sound Database, under data/
Each X.wav has a matching X.txt in the same folder.
EVERY LINE of the .txt is ONE BREATH CYCLE, four columns:

    start_time  end_time  crackles(0/1)  wheezes(0/1)

Label mapping:
    0 0 -> normal
    1 0 -> crackle
    0 1 -> wheeze
    1 1 -> both

Filename format: patientID_recordingIndex_chestLocation_mode_device.wav
Example: 101_1b1_Al_sc_Meditron.wav  ->  patient ID is 101
Patient ID = the number before the FIRST underscore.

## Non-negotiable rules
1. Train/test split MUST be by patient ID, never by file.
   No patient may appear in both train and test. This is the single
   most important correctness rule in the project.
2. SR = 4000 everywhere. The exact same sample rate in training and
   in prediction. A mismatch silently produces garbage output.
3. One breath cycle = one training example. NOT one file.
   ~920 files become ~6900 labelled cycles.
4. The "normal" class heavily dominates. Use class-weighted loss and
   report per-class recall, not just accuracy.
5. Normalise audio identically in training and prediction.

## File plan
- prep.py     audio -> spectrogram PNGs in spectrograms/{train,test}/{class}/
- train.py    ResNet18 transfer learning -> model.pth
- predict.py  the analyze() function below
- app.py      Streamlit dashboard
- metrics.py  confusion matrix, per-class report, plots

## Contract — predict.py must expose exactly this
def analyze(audio_file) -> dict with keys:
    label        str, one of the 4 classes
    confidence   float 0-1
    all_probs    dict of class -> probability
    quality      "good" | "poor"
    cycles       list of (start_sec, end_sec, label)
    salient      (start_sec, end_sec) most influential region
    spec         the spectrogram array
Other files import ONLY this function. Do not change these key names —
app.py is built against them in parallel.

## Explainability approach (keep it simple)
Mute one-second chunks of the audio one at a time, re-predict, and find
which chunk causes the largest confidence drop. That chunk is "salient".
A plain loop. No Grad-CAM library needed.

## Graceful refusal
Before predicting, reject with a clear message if:
- duration < 2 seconds
- mean absolute amplitude < 0.005 (silence)
- top confidence < 0.5  -> "signal quality too low, re-record"
Never crash. Never guess on unusable audio.

## Working style
- Work ONE file at a time. Do not scaffold the whole project at once.
- After writing a script, RUN it and show me the real output.
- Test on 10 files before processing all 920.
- Never invent the dataset format — print a real file and check.
- git commit after each file works. Never commit data/, spectrograms/,
  or *.pth (already in .gitignore).