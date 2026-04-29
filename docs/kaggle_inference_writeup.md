# Kaggle Inference Notebook — Detailed Write-Up

**File:** `notebooks/kaggle_inference.ipynb`
**Purpose:** Run the trained model on Kaggle's hidden test soundscapes and produce a `submission.csv` for scoring.

---

## Why a Separate Notebook?

Training happens on your Mac (locally). But Kaggle competitions require you to submit predictions through a notebook that runs inside Kaggle's own environment. This notebook is that bridge — it takes the trained weights from your Mac, loads them into Kaggle, and runs inference on the hidden test audio files.

The notebook is completely self-contained — no internet, no local files, no external dependencies beyond what Kaggle provides.

---

## Kaggle Environment Constraints

| Constraint | Detail |
|---|---|
| Internet | **Disabled** at submission time — cannot download models or packages |
| GPU | P100 or T4 — much faster than Mac MPS |
| Time limit | 9 hours — inference must complete within this |
| Test data | `test_soundscapes/` is **empty during editing** — only populated at official submission |
| Output | Must write `submission.csv` to `/kaggle/working/` |

---

## Required Inputs

Before running, two datasets must be attached via **+ Add Input** in the Kaggle sidebar:

```
/kaggle/input/
  ├── competitions/
  │     └── birdclef-plus-2026/        ← BirdCLEF+ 2026 competition data
  │           ├── taxonomy.csv
  │           ├── test_soundscapes/
  │           ├── train_soundscapes/
  │           └── sample_submission.csv
  └── datasets/
        └── birdclf2026-weights/       ← your uploaded model checkpoints
              ├── best_fold0.pt
              └── best_fold0_cnn14.pt
```

---

## Cell-by-Cell Walkthrough

### Cell 1 — Package Installation
```python
# !pip install -q librosa timm
```
Commented out by default — `librosa` and `timm` are pre-installed on Kaggle. Uncomment only if a package is missing.

---

### Cell 2 — Imports
```python
import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa, timm
from tqdm.notebook import tqdm
```

| Library | Role in this notebook |
|---|---|
| `numpy` | Audio arrays, spectrogram math, averaging predictions |
| `pandas` | Reading taxonomy.csv, writing submission.csv |
| `torch` | Running the neural network |
| `librosa` | Loading `.ogg` audio files, computing mel spectrograms |
| `timm` | Provides the EfficientNet-B3 architecture |
| `tqdm` | Progress bar while processing soundscapes |

Prints the PyTorch version and device (should show `cuda` on Kaggle GPU).

---

### Cell 3 — Configuration
The most important cell. All paths and settings live here.

**Path detection — auto-finds competition data and weights:**
```python
INPUT_DIR = Path('/kaggle/input')

# Prints the full folder tree so you can see exactly what Kaggle mounted
for root, dirs, files in os.walk(INPUT_DIR): ...

# Finds taxonomy.csv anywhere under /kaggle/input — handles any folder naming
taxonomy_hits = list(INPUT_DIR.rglob('taxonomy.csv'))
COMP_DIR = taxonomy_hits[0].parent

# Finds model checkpoints anywhere under /kaggle/input
weight_hits = list(INPUT_DIR.rglob('best_fold*.pt'))
WEIGHTS_DIR = weight_hits[0].parent
```

Why auto-detect instead of hardcoding? Kaggle changes folder naming across competitions (e.g. `birdclef-2026` vs `birdclef-plus-2026`). Using `rglob` to search for known filenames makes the notebook robust to any naming.

**Audio constants — must match training exactly:**
```python
SAMPLE_RATE = 32_000   # 32kHz — standard for bird audio
WINDOW_SECS = 5        # 5-second chunks (matches submission format)
N_MELS      = 128      # mel frequency bins
N_FFT       = 1024     # FFT window size
HOP_LENGTH  = 320      # ~10ms hop → ~500 time frames per 5s
FMIN        = 50       # ignore frequencies below 50Hz (below bird calls)
FMAX        = 14_000   # ignore frequencies above 14kHz
```

These must be **identical** to what was used during training. If they differ, the spectrogram shape changes and the model produces garbage predictions.

**Inference settings:**
```python
BATCH_SIZE = 64    # how many 5s windows to process at once — larger = faster on GPU
USE_TTA    = True  # test-time augmentation — run each window multiple times with
                   # random masking, average the results for better predictions
N_TTA      = 3     # how many augmented copies to average (original + 3 = 4 total)
```

---

### Cell 4 — Species List
```python
taxonomy_df  = pd.read_csv(TAXONOMY_CSV)
SPECIES_LIST = taxonomy_df['primary_label'].astype(str).tolist()
NUM_CLASSES  = len(SPECIES_LIST)   # 234
```

`taxonomy.csv` defines the **exact order** of the 234 species columns in `submission.csv`. Loading from this file ensures the columns always match what Kaggle expects — even if species were added or removed between competition versions.

---

### Cell 5 — Audio & Spectrogram Utilities

**`audio_to_melspec()`** — converts raw audio into the image-like format the model was trained on:

```
Raw audio waveform          Log-mel spectrogram
[0.002, -0.001, ...]   →   128 rows (frequencies) × ~500 columns (time)
160,000 numbers             each cell = energy at that frequency at that moment
```

Steps:
1. `librosa.feature.melspectrogram()` — applies FFT in sliding windows, maps to mel frequency scale
2. `librosa.power_to_db()` — converts energy values to decibels (log scale, matches human hearing)
3. Normalize to `[0, 1]` — so all spectrograms are on the same scale regardless of recording volume

**`spec_augment()`** — randomly masks frequency and time bands:
```
Before:                    After:
████████████████           ████████████████
████████████████     →     ████████████████
████████████████           ░░░░░░░░░░░░░░░░  ← frequency band zeroed
████████████████           ████████████████
████████████████           ████  ████████    ← time band zeroed
```
Used during TTA (Test-Time Augmentation) to create slightly different versions of each window, then averaging the predictions makes the model more confident and accurate.

---

### Cell 6 — Soundscape Dataset

```python
class SoundscapeDataset(Dataset):
```

Takes one long soundscape file and slices it into non-overlapping 5-second windows:

```
Soundscape recording (e.g. 1 hour long)
│
├── Window 1: seconds  0– 5  →  row_id: BC2026_Test_0001_..._5
├── Window 2: seconds  5–10  →  row_id: BC2026_Test_0001_..._10
├── Window 3: seconds 10–15  →  row_id: BC2026_Test_0001_..._15
└── ...
```

The `row_id` format `{filename_stem}_{end_second}` must match exactly what Kaggle's scoring system expects. Each window is converted to a spectrogram and returned as a PyTorch tensor.

Short clips at the end (if the recording doesn't divide evenly into 5s) are zero-padded to fill the full 5 seconds.

---

### Cell 7 — Model Definitions

Two architectures are defined:

**EfficientNetClassifier**
- Uses `timm.create_model("efficientnet_b3", pretrained=False)` — loads the architecture without downloading weights (no internet)
- `in_chans=1` — accepts single-channel spectrograms (not 3-channel RGB images)
- Custom head: `Linear(1536→512) → ReLU → Dropout → Linear(512→234)`

**CNN14Classifier**
- 6 convolutional blocks, each doubling the channel count: `1→64→128→256→512→1024→2048`
- Global average pooling collapses the spatial dimensions
- Single linear layer to 234 outputs

**`detect_model_type()`** — inspects checkpoint keys to determine which architecture was used:
```python
if 'backbone.conv_stem' in keys → EfficientNet
if 'bn0' in keys               → CNN14
```
This prevents the mismatch error where EfficientNet weights are accidentally loaded into a CNN14 shell (or vice versa).

**`load_model()`** — full loading sequence:
1. `torch.load()` — deserializes the checkpoint from disk
2. `detect_model_type()` — identifies the architecture
3. Builds the correct model class
4. `load_state_dict()` — pours the saved weights into the model
5. `.eval()` — switches off dropout and batch norm training behaviour
6. `.to(device)` — moves model to GPU

---

### Cell 8 — Load Checkpoints

```python
checkpoints = sorted(WEIGHTS_DIR.glob('best_fold*.pt'))
```

Finds all checkpoint files in alphabetical order. For each one, calls `load_model()` which auto-detects the type. All loaded models are collected into a `models` list for ensembling.

Example output:
```
Found 2 checkpoint(s):
  best_fold0.pt
  best_fold0_cnn14.pt
  best_fold0.pt      → detected as efficientnet_b3
  best_fold0_cnn14.pt → detected as cnn14

Ensemble ready: 2 model(s)
```

---

### Cell 9 — Inference Functions

**`predict_batch()`** — runs one batch of windows through all models:

```
Batch of 64 spectrograms
        │
        ├── EfficientNet → sigmoid → probabilities  (64 × 234)
        │     ├── TTA pass 1 → probabilities
        │     ├── TTA pass 2 → probabilities
        │     └── TTA pass 3 → probabilities
        │
        └── CNN14 → sigmoid → probabilities  (64 × 234)
              ├── TTA pass 1 → probabilities
              ├── TTA pass 2 → probabilities
              └── TTA pass 3 → probabilities

Average all 8 sets of probabilities → final (64 × 234)
```

Total versions averaged per batch with 2 models and N_TTA=3:
`2 models × (1 original + 3 TTA) = 8 predictions averaged`

**`run_inference()`** — outer loop over all soundscape files:
1. Finds all `.ogg/.flac/.wav/.mp3` files in the target directory
2. For each file, creates a `SoundscapeDataset` and a `DataLoader`
3. Calls `predict_batch()` on each batch
4. Collects all `row_id` strings and probability arrays
5. Assembles everything into a Pandas DataFrame

---

### Cell 10 — Run Inference (with Fallback)

```python
if len(test_files) == 0:
    # Interactive mode — test_soundscapes/ is empty, use train_soundscapes/ to test pipeline
    INFER_DIR = COMP_DIR / 'train_soundscapes'
else:
    # Official submission — real test files present
    INFER_DIR = TEST_AUDIO_DIR
```

**Why the fallback?**
Kaggle's `test_soundscapes/` folder is intentionally empty during interactive notebook editing. The real test audio only appears when Kaggle re-runs the notebook as an official submission (via Save Version → Commit). Using `train_soundscapes/` as a fallback lets you verify the entire pipeline end-to-end without waiting for a full submission.

---

### Cell 11 — Write Submission

```python
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
expected_columns = sample_sub.columns.tolist()
submission = predictions_df[expected_columns]   # reorder to match exactly
submission.to_csv('/kaggle/working/submission.csv', index=False)
```

Reads `sample_submission.csv` to get the exact column order Kaggle expects, then reorders the predictions DataFrame to match. This guards against any accidental column ordering differences.

---

### Cell 12 — Sanity Checks

Three assertions before considering the notebook done:

| Check | What it catches |
|---|---|
| `isnull().sum() == 0` | Missing predictions for any window or species |
| `min() >= 0` and `max() <= 1` | Probabilities outside valid range (should never happen after sigmoid) |
| `columns == expected_columns` | Column order mismatch vs sample submission |

If all pass, prints a summary:
```
All checks passed ✓
  Rows    : 4800      (number of 5s windows across all test soundscapes)
  Columns : 235       (row_id + 234 species)
  Prob range: [0.0012, 0.9831]
```

---

## How to Submit

### Interactive testing (now)
Run all cells → verify no errors → check the output in `/kaggle/working/`

### Official submission
1. Turn **internet OFF** in Settings
2. Click **Save Version** → **Save & Run All (Commit)**
3. Kaggle re-runs the notebook with real test data populated
4. Score appears on the leaderboard within minutes

---

## Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: taxonomy.csv` | Competition data not added | Click `+ Add Input` → Competition Datasets → BirdCLEF+ 2026 |
| `AssertionError: No BirdCLEF folder` | Same as above | Same fix |
| `AssertionError: No best_fold*.pt` | Weights not uploaded or not added | Upload `.pt` files to Kaggle dataset, add via `+ Add Input` |
| `RuntimeError: Missing keys in state_dict` | Model type mismatch | Fixed — `detect_model_type()` handles this automatically |
| `ValueError: need at least one array` | `test_soundscapes/` empty | Fixed — falls back to `train_soundscapes/` automatically |
| `AssertionError: Column mismatch` | Species list order wrong | Ensure `taxonomy.csv` is from the competition data, not a local copy |
