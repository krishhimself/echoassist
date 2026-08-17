# EchoAssist — Complete Technical Breakdown

**Problem Statement PS-S01 | Healthcare Domain | 24-Hour Hackathon**
Lung sound classification with explainability and graceful degradation.

---

## Part 1 — The Problem We're Solving

A doctor places a stethoscope on a patient's chest and listens to them breathe. From that sound alone they judge whether the lungs are healthy or whether something is wrong.

The weakness is that this is **one person's ears making a subjective judgement**. Two trained doctors listening to the same recording can reach different conclusions. Subtle sounds get lost in ambient noise. And the judgement leaves no record — it cannot be re-examined, compared over time, or audited.

Digital stethoscopes solved the *recording* problem. Nobody solved the *interpretation* problem.

**What we built:** a system that takes a breathing recording and classifies it into one of four categories, states how confident it is, and shows which part of the audio drove that decision.

### The four classes

| Class | What it means | Sound character |
|---|---|---|
| **normal** | Healthy breathing | Smooth, broadband airflow |
| **crackle** | Discontinuous popping | Very short clicks, 5–20 ms, usually on inhale |
| **wheeze** | Continuous whistling | Sustained high-frequency tone, usually on exhale |
| **both** | Crackles *and* wheezes present | Both patterns in the same breath |

### Our hard scope limit

We describe the **sound**. We never output a diagnosis. "This recording contains wheezing" is a legitimate claim about acoustics. "This patient has asthma" is a medical diagnosis and is outside what our system is permitted to assert. This boundary is written into the problem statement and we respect it in every output, including the PDF report footer.

---

## Part 2 — The Core Idea (understand this and everything else follows)

> **We convert sound into a picture, then perform image recognition on the picture.**

This single reframe is what makes the entire project tractable.

A **spectrogram** is a picture of sound:

- **Horizontal axis** = time
- **Vertical axis** = pitch (low frequencies at bottom, high at top)
- **Brightness** = how much energy exists at that pitch at that moment

Once audio becomes an image, our problem becomes "sort these images into four categories" — which is the single most well-solved problem in machine learning. Decades of research, thousands of tutorials, and freely available pre-trained models all apply directly.

### What the classes look like as pictures

- **Wheeze** → long bright *horizontal* streaks (a sustained tone holds one pitch across time)
- **Crackle** → short bright *vertical* specks (a click contains all frequencies for an instant)
- **Normal** → diffuse, textured, no sharp structure
- **Both** → horizontal streaks and vertical specks together

This is genuinely visible to the human eye. Open `spectrograms/train/wheeze/` next to `spectrograms/train/normal/` and you can tell them apart yourself. That visual check was our first validation that the pipeline worked.

---

## Part 3 — The Dataset

**ICBHI 2017 Respiratory Sound Database** — 920 real patient recordings, publicly available on Kaggle, no credentialing required.

### The critical structural insight

Each `X.wav` has a matching `X.txt` in the same folder. **Every line of the `.txt` is one breath cycle**, with four columns:

```
start_time   end_time   crackles(0/1)   wheezes(0/1)
```

Real example from `101_1b1_Al_sc_Meditron.txt`:

```
0.036    0.579    0    0
0.579    2.450    0    0
2.450    3.893    0    0
```

Label mapping:

| crackles | wheezes | class |
|---|---|---|
| 0 | 0 | normal |
| 1 | 0 | crackle |
| 0 | 1 | wheeze |
| 1 | 1 | both |

### Why this matters enormously

Most teams treat one file as one training example — 920 examples total, which is far too few to train anything.

Because the annotations mark every individual breath, **we treat one breath cycle as one training example.** That turns 920 files into **6,857 labelled examples** — a 7.5× increase in training data, obtained for free from data already in the download.

It also gives us the per-segment timeline feature at no extra cost, since we already know where each breath starts and ends.

### Filename structure

```
101_1b1_Al_sc_Meditron.wav
 |   |   |   |     |
 |   |   |   |     device used
 |   |   |   acquisition mode
 |   |   chest location
 |   recording index
 patient ID  <-- the number before the first underscore
```

That patient ID is the single most important field in the dataset, for reasons explained in Part 6.

### Actual class distribution we produced

| Class | Train | Test | Total | Share |
|---|---|---|---|---|
| normal | 2,780 | 828 | 3,608 | 53% |
| crackle | 1,196 | 665 | 1,861 | 27% |
| wheeze | 645 | 237 | 882 | 13% |
| both | 351 | 155 | 506 | 7% |
| **total** | **4,972** | **1,885** | **6,857** | |

Patients: **100 train / 26 test**, verified disjoint.

---

## Part 4 — Libraries and What Each One Actually Does

| Library | Role in our project | Specifically |
|---|---|---|
| **librosa** | Audio processing | `librosa.load()` reads .wav at a chosen sample rate. `librosa.feature.melspectrogram()` converts waveform to spectrogram. `librosa.power_to_db()` converts to decibel scale. |
| **numpy** | Array mathematics | Slicing breath cycles out of the waveform, padding to fixed length, peak normalisation, muting chunks for explainability. |
| **matplotlib** | Image output | `imsave()` writes each spectrogram array as a PNG with the `magma` colormap. Also draws the confusion matrix and metric charts. |
| **PyTorch (torch)** | The neural network | Holds the model, computes the loss, runs backpropagation, updates weights, saves/loads `model.pth`. |
| **torchvision** | Pre-built vision tools | Provides pre-trained ResNet18, and `ImageFolder` which reads a folder-per-class directory structure into a labelled dataset automatically. |
| **scikit-learn** | Evaluation | `classification_report()` gives precision/recall/F1 per class in one call. `confusion_matrix()` shows exactly which classes get mistaken for which. |
| **Streamlit** | The dashboard | Turns plain Python into a web interface. No HTML, CSS or JavaScript written anywhere in this project. |
| **soundfile** | Writing .wav files | Used to generate the deliberately bad test files (silent, too-short) for verifying graceful refusal. |

### Environment

```
Windows, PowerShell
Python 3.14, virtual environment at .venv
librosa 1.0.0
torch 2.13.0+cpu    <-- CPU ONLY, no GPU available
```

**The CPU constraint drove real design decisions.** Standard image tutorials resize everything to 224×224 pixels because that is what ImageNet models expect. We deliberately did *not* — our spectrograms stay at 64×157. On CPU this is the difference between roughly 2 minutes and roughly 20 minutes per epoch. It also loses us nothing, because a spectrogram is not a photograph and has no fine detail to preserve at high resolution.

---

## Part 5 — The Code, File by File

### Architecture

```
data/ICBHI_final_database/          920 .wav + 920 .txt
          |
          |  prep.py
          v
spectrograms/{train,test}/{class}/  6,857 PNG images
          |
          |  train.py
          v
      model.pth                     trained ResNet18 weights
          |
          |  predict.py  <-- the single interface everything else uses
          v
   analyze(audio) -> dict
          |
    +-----+-----+
    |           |
  app.py     metrics.py
 (dashboard) (evaluation)
```

---

### `prep.py` — audio into training images

**Purpose:** turn 920 recordings into 6,857 labelled spectrogram images, correctly split by patient.

**Constants and why each value:**

```python
SR            = 4000    # resample everything to 4 kHz
CYCLE_SECS    = 5.0     # every cycle padded/trimmed to exactly 5 s
CYCLE_SAMPLES = 20000   # 4000 x 5
MIN_SECS      = 0.5     # discard cycles shorter than this
N_MELS        = 64      # 64 frequency bands = image height
SEED          = 42      # fixed seed so the split is reproducible
```

**Why 4000 Hz** when the originals are 44,100 Hz? Lung sounds live below ~2000 Hz. By the Nyquist theorem, a 4000 Hz sample rate captures everything up to 2000 Hz. Everything above that is speech, room noise and electrical hum — discarding it is *itself a noise-reduction step*, and it makes every downstream computation 11× cheaper.

**Why a fixed 5 seconds?** A neural network needs consistent input dimensions. Real cycles vary from 0.5 s to 5 s+, so we pad short ones with zeros and trim long ones.

**Step-by-step flow:**

1. **Collect patients.** Read all 920 filenames, extract the patient ID from each (`name.split("_")[0]`), build a set of unique patients.
2. **Split by patient.** Shuffle the patient list with the fixed seed, take 80% for train and 20% for test. Then `assert train_pids.isdisjoint(test_pids)` — the code refuses to run if any patient appears in both.
3. **For each .wav:**
   - `librosa.load(path, sr=4000)` — read and resample
   - `y = y / np.max(np.abs(y))` — peak normalisation, so loud and quiet recordings are treated equally
4. **For each line of the matching .txt:**
   - Parse `start_time`, `end_time`, `crackles`, `wheezes`
   - Skip if the cycle is under 0.5 s
   - Slice: `y[int(start*SR) : int(end*SR)]`
   - Pad with zeros or trim to exactly 20,000 samples
   - `melspectrogram(y=cycle, sr=4000, n_mels=64, n_fft=512, hop_length=128)`
   - `power_to_db(mel, ref=np.max)` — convert to decibels
   - `imsave(..., cmap='magma')` into `spectrograms/{split}/{class}/`
5. **Report** cycles per class per split.

**Output shape:** `(64, 157, 4)` — 64 frequency bands tall, 157 time frames wide, 4 channels (RGBA from the colormap).

---

### The `n_fft` decision — the subtlest and most important tuning choice

Our first run produced spectrograms only 40 frames wide. That looked fine but was quietly fatal.

librosa's default `n_fft` is 2048 samples. At 4000 Hz that is a **512-millisecond analysis window** — every output pixel summarises half a second of audio.

**A crackle lasts 5–20 milliseconds.** Inside a 512 ms window it contributes roughly 2% of the energy and is smeared into invisibility.

The consequence would have been a model **structurally incapable** of detecting crackles — one of our four classes, and 27% of our data. Wheezes would still work fine because they are sustained, so overall accuracy would have looked acceptable while crackle recall stayed near zero. The bug would have been invisible in the headline number.

**Fix:** `n_fft=512, hop_length=128`. That gives a 128 ms analysis window stepping every 32 ms, and 157 frames instead of 40.

> **The general lesson:** signal-processing defaults are tuned for music and speech. Medical acoustics have different timescales. Always check that your representation can physically resolve the phenomenon you are trying to detect.

**These four values — `SR`, `N_MELS`, `n_fft`, `hop_length` — must be byte-identical in `prep.py` and `predict.py`.** If they differ, the model receives data on a completely different scale than it trained on and outputs confident nonsense, while every line of code appears to run correctly.

---

### `train.py` — teaching the model

**Purpose:** train a classifier on the spectrogram images and save the best version.

**Configuration:**

```python
model      = ResNet18(pretrained=True)   # replace final layer with 4 outputs
loss       = CrossEntropyLoss(weight=class_weights)
optimizer  = Adam(lr=1e-4)
batch_size = 32
epochs     = 10
```

**Transfer learning — what "pretrained" means.** ResNet18 has already been trained on millions of photographs. In that process its early layers learned generic visual primitives: edges, textures, gradients, repeating patterns. Those primitives are *equally useful* on spectrograms — a wheeze is a horizontal edge, a crackle is a vertical one.

So we keep all of that learned machinery and replace only the final classification layer (originally 1000 photo categories) with a fresh 4-output layer. We are teaching an experienced pattern-recogniser a new vocabulary, not teaching a newborn to see. On our data size this is the difference between a usable model and a useless one.

**Class weighting.** Our data is 53% normal, 7% both. Untreated, the model learns that always guessing "normal" is a great strategy — 53% accuracy for zero understanding.

Fix: weight each class inversely to its frequency (`weight = 1/count`, normalised). Getting a rare `both` example wrong now costs the model far more than getting a `normal` example wrong, so it is forced to actually learn the rare classes.

**Selection metric — the choice that keeps us honest.** We save the best checkpoint by **macro-average recall**, not accuracy.

- *Accuracy* = what fraction of all predictions were correct. Dominated by the majority class. An always-normal model scores 44% on our test set.
- *Recall (per class)* = of all the true wheezes, what fraction did we catch?
- *Macro-average recall* = the plain average of the four per-class recalls, weighting every class equally regardless of size.

An always-normal model scores macro-recall 0.25 (it gets normal perfectly and the other three at zero). **Macro-recall cannot be gamed by ignoring rare classes.** That is exactly why we use it.

---

### `predict.py` — the single interface

**Purpose:** one function that everything else in the project calls.

```python
def analyze(audio_file) -> dict:
    return {
        "label":       str,     # one of the 4 classes
        "confidence":  float,   # 0 to 1
        "all_probs":   dict,    # class -> probability
        "quality":     str,     # "good" or "poor"
        "cycles":      list,    # [(start_sec, end_sec, label), ...]
        "salient":     tuple,   # (start_sec, end_sec) most influential region
        "spec":        ndarray  # the spectrogram, for display
    }
```

**Why this design mattered so much in a 24-hour build.** We wrote this function first with **hardcoded fake values** and no model at all. That let the dashboard developer build the entire UI in hour 2, while training had not even started. When the real model was ready, we swapped the function's internals and every consumer kept working untouched.

**This is the single most valuable structural decision in the project.** Without it, the UI developer sits idle waiting for the model, then panics late. Define the contract early, let both sides build against it in parallel.

**Three responsibilities inside the function:**

**1. Quality gating (graceful refusal).** Before any prediction:

```
duration < 2 seconds                 -> refuse
mean(abs(audio)) < 0.005 (silence)   -> refuse
top confidence < 0.5                 -> "signal quality too low, re-record"
```

This is the problem statement's "graceful handling of edge cases" requirement, but framed as a *feature* rather than a `try/except`. Almost every competing team builds a model that always emits an answer, even on pure static. Ours can decline. In a clinical context a confident wrong answer is worse than no answer, and being able to say "I don't know" is the mature behaviour.

**2. Classification.** Slide a window across the audio, spectrogram each window, run the model, average the probabilities. Per-window predictions become the segment timeline.

**3. Explainability by occlusion.** The requirement is to show *which part of the signal drove the classification*. The standard approach is Grad-CAM, which needs extra libraries and gradient plumbing. We used something simpler and equally valid:

```
predict on the full audio       -> baseline confidence
for each 1-second chunk:
    mute that chunk (set samples to 0)
    predict again
    drop = baseline_confidence - new_confidence
the chunk with the largest drop is the salient region
```

**Why this is legitimate, not a shortcut:** if muting a second of audio causes confidence to collapse, the model was demonstrably relying on that second. If muting it changes nothing, the model was ignoring it. It is a direct causal measurement of importance — arguably more interpretable than a gradient-based saliency map, and it is a plain `for` loop with no dependencies.

Our verification that it works: on a real file, it returned 6.0–7.0 s. A broken implementation almost always returns 0.0 s or the final chunk every time.

---

### `app.py` — the dashboard

Streamlit renders Python as a web page. Components:

- Sample selector and file upload
- Audio player for the original recording
- Predicted label and confidence, prominently
- All four class probabilities as a bar chart — this shows *uncertainty*, not just the winner
- The mel spectrogram with the salient region shaded
- **A button that plays only the salient second**
- Segment timeline, colour-coded by predicted class

That play-the-salient-second button is the strongest single moment in our demo. A heatmap asks the judge to trust a visualisation. Playing the exact 0.4 seconds the model reacted to lets them **hear** what it heard.

---

### `metrics.py` — proving it works

Uses scikit-learn to produce accuracy, and per-class precision, recall and F1, plus a confusion matrix saved as a PNG.

**Reading the three metrics:**

- **Precision** — when we said "wheeze", how often were we right? Low precision = false alarms.
- **Recall** — of all real wheezes, how many did we catch? Low recall = missed cases.
- **F1** — the harmonic mean of the two; a single balanced number.

In a clinical screening context, **recall matters more than precision**. A missed abnormality is more dangerous than a false alarm that a doctor then dismisses.

The confusion matrix is more informative than any single number, because it shows *which* classes get confused. Ours shows `both` being predicted as `normal` (77 times) or `wheeze` (45 times) — telling us the model detects that something is there but cannot reliably see two phenomena at once.

---

## Part 6 — The Decisions That Actually Matter

### Decision 1: Split by patient, never by file

**The rule:** no patient may have recordings in both the training and test sets.

**Why:** several recordings come from the same patient, on the same equipment, in the same room, with the same body habitus and background noise. If patient 112 appears in both sets, the model can recognise *"this is patient 112's recording setup"* rather than *"this is a wheeze"*, and score brilliantly on the test set while having learned nothing transferable.

This is data leakage, and it is the most common serious error in medical ML.

**How we enforce it:** the split happens over patient IDs, before any file is touched, followed by `assert train_pids.isdisjoint(test_pids)`. The program refuses to run if the condition fails.

**The cost:** this decision alone probably costs us 20–30 percentage points of headline accuracy compared to a file-level split. **It is the single most important correctness property of the project.** A published-looking 90% obtained by leaking patients is worth nothing; our 49% is real.

There was also a subtle version of this trap we had to avoid: our 10-file test run computed its split over only 4 patients, so a patient assigned to `train/` there could belong in `test/` in the full run. Because the PNGs persist on disk, not wiping `spectrograms/` before the full run would have reintroduced leakage through the back door. We deleted the folder first.

### Decision 2: Macro-recall as the selection metric

Covered in Part 5. The short version: accuracy can be gamed by the majority class, macro-recall cannot.

### Decision 3: Tune the spectrogram for the shortest phenomenon

Covered in Part 5. If your representation cannot physically resolve the thing you want to detect, no amount of model capacity will rescue it.

### Decision 4: Contract-first parallel development

Define the interface, build both sides against it simultaneously. This is what let four workstreams proceed without blocking each other.

### Decision 5: Never overwrite a working artifact

When we retrained with augmentation, output went to `model2.pth`, not `model.pth`. We had a functioning end-to-end demo built on the original weights. Overwriting it would have destroyed our fallback if the retrain came out worse.

**General rule for time-boxed builds: always keep a known-good version.**

---

## Part 7 — Results

### Baseline: `model.pth` (10 epochs, no augmentation)

**Accuracy 49.3% | Macro-recall 0.377** on 1,885 held-out cycles from 26 unseen patients.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal | 0.511 | 0.611 | 0.556 | 828 |
| crackle | 0.537 | 0.513 | 0.525 | 665 |
| wheeze | 0.335 | 0.283 | 0.307 | 237 |
| both | 0.271 | 0.103 | 0.150 | 155 |

Wall-clock training time: 23.3 minutes on CPU.

### Diagnosis of the weakness

Train loss dropped to 0.09 while test macro-recall plateaued at 0.38 — a large gap, which is the textbook signature of **overfitting**. The model had memorised the training set rather than learning generalisable features. 6,857 images is small for a network with 11 million parameters.

Crucially, **more epochs would have made this worse, not better.** More passes over data the model has already memorised deepens memorisation. The correct lever was not more training but more *variety* in training.

### Improvement: `train2.py` with augmentation

Changes made:

- **SpecAugment** — randomly mask 1–2 horizontal and 1–2 vertical bars in each training image. The model sees a slightly different image every epoch, so it cannot memorise, and it is forced to recognise a wheeze even when part of it is hidden. This is the standard treatment for exactly our symptom.
- **Random time shift** — roll the spectrogram ±10 frames.
- **Weight decay 1e-4** — penalise large weights, a general regulariser.
- **Rare classes boosted 2×** beyond the inverse-frequency weighting.

Result at epoch 11 of 20: **macro-recall 0.400**, and critically, **train loss 0.69 instead of 0.09** — proof the model was still learning rather than memorising. The `both` class improved from **0.103 → 0.239**, more than doubling on our weakest category.

### Why ~50% is a good result here, and how to say so

1. **Random guessing on 4 classes is 25%.** We are at nearly double that.
2. **Always-predicting-normal scores 44% accuracy** on our test set, and macro-recall 0.25. Our macro-recall of 0.40 cannot be reached that way.
3. **Our split is patient-independent.** Reports of 85–95% on ICBHI almost always come from file-level splits, which leak the same patient into both sets. Those numbers are not comparable to ours.
4. **This is a recognised hard benchmark.** Patient-independent 4-class ICBHI classification is genuinely difficult, and published research using months of work and GPU clusters reports scores in a broadly comparable range.

**How to present it:** *"49% accuracy, 0.40 macro-recall, against a 25% random baseline. We report macro-recall rather than accuracy because always predicting 'normal' would score 44% and be clinically worthless. Our train/test split is patient-independent, which is why our figure is lower than numbers you may see elsewhere — file-level splits leak the same patient into both sets and inflate results substantially."*

Then show the confusion matrix and name `both` as the weak class, with the reason: only 7% of the data, and it requires detecting two phenomena simultaneously.

A judge who knows this field will trust a well-explained 49% far more than an unexplained 94%.

---

## Part 8 — Concepts Glossary

**Sample rate (Hz)** — audio samples captured per second. 44,100 Hz is CD quality. We downsample to 4,000 Hz because lung sounds contain no useful information above 2,000 Hz.

**Nyquist theorem** — a sample rate of N Hz can faithfully represent frequencies up to N/2 Hz. This is why 4,000 Hz suffices for content below 2,000 Hz.

**Spectrogram** — a 2D image of sound: time on one axis, frequency on the other, brightness as energy.

**Mel scale** — a frequency scale spaced according to human pitch perception rather than raw Hz. Human hearing distinguishes low frequencies more finely than high ones, and the mel scale reflects that, so mel bands carry more perceptually relevant information than linear ones.

**Decibel (dB) scale** — a logarithmic loudness scale. Sound energy spans an enormous range; taking the log compresses it into something a neural network can learn from.

**n_fft** — the number of samples in each analysis window. Larger = better frequency precision but worse time precision. This trade-off is fundamental and unavoidable, and choosing the wrong side of it is what nearly cost us the crackle class.

**hop_length** — how far the analysis window advances each step. Determines how many time frames the output has.

**Epoch** — one complete pass through the training data.

**Batch size** — how many examples the model processes before updating its weights. 32 is a standard compromise between speed and stability.

**Learning rate** — how large a step to take when updating weights. Too high and training diverges; too low and it never converges. 1e-4 is a common starting point for fine-tuning.

**Transfer learning** — starting from a model already trained on a different, larger task and adapting it. Essential when your own dataset is small.

**Overfitting** — memorising the training data instead of learning generalisable patterns. Diagnosed by a large gap between training performance and test performance.

**Regularisation** — any technique that makes memorisation harder: weight decay, dropout, data augmentation.

**Data augmentation** — creating modified copies of training data so the model sees more variety and cannot memorise.

**SpecAugment** — augmentation designed for spectrograms: randomly mask out horizontal (frequency) and vertical (time) bands.

**Class imbalance** — when some classes have far more examples than others. Untreated, models learn to ignore the rare ones.

**Class weighting** — making errors on rare classes cost more in the loss function, forcing the model to attend to them.

**Data leakage** — when information from the test set influences training. Our patient-split rule exists entirely to prevent this.

**Precision / Recall / F1** — precision is how often positive predictions are right; recall is how many real positives were caught; F1 is their harmonic mean.

**Macro-average** — the plain average across classes, weighting each equally regardless of size. Contrast with micro-average, which weights by class frequency and is therefore dominated by the majority class.

**Confusion matrix** — a grid showing which true classes get predicted as which. More diagnostically useful than any single scalar metric.

**Occlusion analysis** — hiding parts of the input and measuring how much the prediction changes, to identify which parts mattered.

**Softmax** — converts raw model outputs into probabilities summing to 1.

---

## Part 9 — Requirements Status

| Problem statement requirement | How we satisfied it |
|---|---|
| Noise-reduction pipeline | Downsampling to 4 kHz discards out-of-band noise; peak normalisation equalises volume |
| Feature extraction, core acoustic markers | Mel spectrograms per breath cycle, with `n_fft` tuned to resolve crackles |
| Classifier, normal + 2–3 abnormal, with confidence | ResNet18, 4 classes, softmax probabilities exposed |
| Explainability view | Occlusion analysis identifies and displays the most influential region, with audio playback |
| Working dashboard, end-to-end | Streamlit app: upload/select, predict, visualise, play salient region |
| Evaluation metrics across all classes | Accuracy, per-class precision/recall/F1, confusion matrix |
| Graceful edge-case handling | Duration, silence and confidence gates; verified on silent and 1-second files without crashing |

All seven verified by manual end-to-end testing.

### Remaining work

- Swap in `model2.pth` after comparing full per-class metrics, not just macro-recall
- Curated 8-file sample gallery with friendly labels, replacing the 920-item dropdown
- Use real `.txt` breath boundaries in the timeline for annotated files, showing predicted vs ground truth per cycle
- PDF report export with the "not a diagnostic device" footer
- Accuracy-versus-noise curve — inject noise at 20/15/10/5/0 dB SNR and plot degradation, turning "degrades gracefully" from a claim into a measurement
- Test-time averaging over overlapping windows
- Model size and CPU inference latency, for the edge-deployment argument

---

## Part 10 — The Demo

### Seven beats

1. **The problem.** Two trained doctors, same recording, different conclusions.
2. **Play a lung sound aloud.** Ask the room: can you tell? They cannot.
3. **Run it.** Label, confidence, all four probabilities, timeline.
4. **Press play on the salient second.** "This is what the model heard."
5. **Feed it silence.** It refuses. *"It knows when it doesn't know."*
6. **The metrics.** Patient-level split, confusion matrix, honest per-class recall.
7. **Close on scope.** "Acoustic decision support, not diagnosis."

### The three points that show understanding

Lead with these if time is short:

1. **Sound becomes a picture, then image recognition.** The reframe that made the project tractable.
2. **We split by patient, not by file.** Explain the leakage it prevents and that it costs us headline accuracy. This is the strongest signal of competence available to us.
3. **We tuned `n_fft` to resolve crackles.** The default 512 ms window would have made one of our four classes physically undetectable.

### The edge-deployment argument

Our model is small and runs on CPU. Report its file size and single-prediction latency, then: *"this runs on a Raspberry Pi with an inexpensive microphone, in a clinic with no internet and no specialist."*

Every other team demonstrates a laptop notebook. A believable deployment story in the setting where auscultation expertise is actually scarce is worth more than a marginally better accuracy figure.

### Contingency

Record a backup video of a successful run. Live demos fail for reasons unrelated to code quality — projector handshakes, dead speakers, unscheduled OS updates. Freeze features well before the deadline and rehearse instead.

---

## Part 11 — Transferable Lessons

1. **Reframe until the problem becomes a solved problem.** Audio classification is hard; image classification is solved. Convert one into the other.
2. **Read the data format from the actual file.** Never infer it. Our first action was printing a real `.txt` and verifying every column.
3. **Check that your representation can resolve your phenomenon.** Signal-processing defaults are tuned for music and speech, not medical acoustics.
4. **Define interfaces before implementations.** It is what makes parallel work possible.
5. **Choose metrics that cannot be gamed.** Accuracy on imbalanced data flatters useless models.
6. **Prevent leakage structurally, with an assertion.** Not with a comment, not with good intentions.
7. **Never overwrite a working artifact.** Always keep a fallback.
8. **Diagnose before treating.** Our overfitting needed augmentation, not more epochs. More epochs would have made it worse.
9. **Verify end-to-end by hand.** Seven passing components that have never been run in sequence are not a working system.
10. **Report honest numbers with context.** A well-explained 49% is more credible than an unexplained 94%.
