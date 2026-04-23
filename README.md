# Pantanal Wildlife Audio Classification
### BirdCLEF 2026 — Kaggle Competition

Automated identification of **234 wildlife species** from audio recordings collected across the Pantanal wetlands (Brazil). Species include birds (Aves), frogs (Amphibia), insects, mammals, and reptiles.

---

## Competition Summary

| Property | Detail |
|---|---|
| Task | Multi-label audio classification |
| Classes | 234 species (162 birds, 35 amphibians, 28 insects, 8 mammals, 1 reptile) |
| Input | `.ogg` soundscape recordings, continuous audio |
| Output | Per-species probability for each 5-second window |
| Metric | Mean ROC-AUC across species |
| Submission unit | `row_id = {soundscape_stem}_{end_second}` |

---

## High-Level System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
│                                                                      │
│  train_audio/          train_soundscapes/         taxonomy.csv       │
│  35,549 short clips    10,658 labeled windows     234 species        │
│  (iNat + Xeno-canto)   (same dist as test)        ordered list       │
└────────────┬──────────────────┬──────────────────────┬──────────────┘
             │                  │                      │
             ▼                  ▼                      ▼
┌────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                                  │
│                                                                     │
│   Random 5s crop        5s window at             Species → int      │
│   (train: random,       labeled timestamp         index mapping     │
│    val: start=0)        (start/end from CSV)      (234 classes)     │
│            │                  │                                     │
│            └──────────────────┘                                     │
│                       │                                             │
│                       ▼                                             │
│            Audio waveform  (32kHz, mono, 160,000 samples)          │
│                       │                                             │
│                       ▼                                             │
│         Log-Mel Spectrogram  (128 mel bins × ~500 time frames)     │
│            n_fft=1024, hop=320, fmin=50Hz, fmax=14kHz              │
│            Normalized to [0, 1] per clip                           │
└───────────────────────────┬────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    AUGMENTATION (training only)                    │
│                                                                    │
│   SpecAugment          Mixup              Background Noise         │
│   ──────────           ──────             ────────────────         │
│   Mask random          Blend 2            Add rain/wind/river      │
│   freq bands           spectrograms       at random SNR (5–20dB)   │
│   & time bands         + interpolate      applied 30% of time      │
│   (2×freq, 2×time)     labels (α=0.4)                              │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      MODEL ARCHITECTURES                           │
│                                                                    │
│   Option A: EfficientNet-B3          Option B: CNN14               │
│   ─────────────────────────          ────────────────              │
│   ImageNet pretrained                AudioSet pretrained           │
│   in_chans=1 (spectrogram)           (PANNs checkpoint)            │
│   Global avg pool                    6 × Conv blocks               │
│   → Linear(512) → ReLU              → Global avg pool              │
│   → Linear(234)                      → Linear(234)                 │
│                                                                    │
│   Output: logits (B, 234)  →  sigmoid at inference                │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                       TRAINING STRATEGY                            │
│                                                                    │
│   Loss: Focal BCE (γ=2.0)            Why: down-weights easy       │
│         + per-class pos_weight        negatives, critical with     │
│         clamped to [1, 10]            234 classes + imbalance      │
│                                                                    │
│   Optimizer: AdamW (lr=1e-3, wd=1e-4)                             │
│   Schedule:  Linear warmup (2 ep) → Cosine decay                  │
│   Mixed precision: float16 via torch.autocast                     │
│                                                                    │
│   Cross-validation: 5-fold, stratified by primary_label           │
│   Train set = clips + soundscape windows (ConcatDataset)          │
│   Val set   = held-out clips only                                  │
│                                                                    │
│   Checkpoint: best val ROC-AUC per fold → best_fold{N}.pt         │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                             │
│                                                                    │
│  Test soundscape (long .ogg)                                       │
│          │                                                         │
│          ▼                                                         │
│  Slide non-overlapping 5s windows                                  │
│  row_id = "{stem}_{end_second}"                                    │
│  e.g. BC2026_Test_0001_..._5, ..._10, ..._15 ...                  │
│          │                                                         │
│          ▼                                                         │
│  Ensemble (5 fold checkpoints, equal weights)                      │
│          │                                                         │
│          ▼                                                         │
│  Test-Time Augmentation (TTA)                                      │
│  Average over original + 3 SpecAugmented versions                 │
│          │                                                         │
│          ▼                                                         │
│  Probability matrix  (N_windows × 234)                            │
│          │                                                         │
│          ▼                                                         │
│  submission.csv  ──► Kaggle                                        │
└───────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pantanal-audio-classification/
├── src/
│   ├── dataset.py       Audio loading, spectrogram conversion, all dataset classes
│   ├── model.py         EfficientNet-B3 and CNN14 classifiers, EnsembleModel
│   ├── train.py         Full training loop — focal loss, CV folds, checkpointing
│   └── inference.py     Sliding-window inference, TTA, submission CSV generation
├── configs/
│   └── default.json     All hyperparameters in one file
├── checkpoints/         Saved model weights (best per fold)
├── requirements.txt
└── README.md

../birdclef-2026-2/      (competition data, sibling directory)
├── taxonomy.csv         234 species with primary_label codes (defines column order)
├── train.csv            35,549 clip metadata rows
├── train_audio/         Short clips, organized by primary_label subdirectory
├── train_soundscapes/   Long recordings used as labeled training soundscapes
├── train_soundscapes_labels.csv  1,478 labeled 5s windows (same format as test)
├── test_soundscapes/    Hidden at runtime — populated by Kaggle on rerun
└── sample_submission.csv  Shows expected row_id format and column order
```

---

## Data Deep-Dive

### Species Breakdown (234 total)

| Class | Count | Notes |
|---|---|---|
| Aves (birds) | 162 | eBird 6-letter codes, e.g. `rubthr1`, `houspa` |
| Amphibia (frogs) | 35 | iNat taxon IDs, e.g. `1176823` |
| Insecta | 28 | includes `47158son01`–`47158son25` (25 sonotypes of one species) |
| Mammalia | 8 | iNat taxon IDs |
| Reptilia | 1 | `116570` = Southern Spectacled Caiman |

### Class Balance

- **Top species**: ~499 clips (well-represented)
- **Bottom species**: 1–3 clips (extremely rare)
- Addressed via: focal loss + per-class positive weighting + balanced sampling

### Key Label Fields in train.csv

| Column | Example | Notes |
|---|---|---|
| `primary_label` | `rubthr1` | The definite species in the clip |
| `secondary_labels` | `['houspa', 'banana']` | Background species — treated as soft labels (weight 0.4) |
| `filename` | `rubthr1/XC123456.ogg` | Relative path under `train_audio/` |
| `rating` | `4.5` | Xeno-canto quality rating; can filter on this |

### Why train_soundscapes_labels.csv Matters

The test data consists of long soundscape recordings. Training on short clean clips creates a domain mismatch. The 1,478 labeled windows from `train_soundscapes_labels.csv` are in the **same acoustic environment as the test set**, making them disproportionately valuable. Both datasets are combined during training via `ConcatDataset`.

---

## Model Design Rationale

### Pretrained Model Origins

#### ImageNet pretrained — EfficientNet-B3

| Component | Creator | Year | Notes |
|---|---|---|---|
| **ImageNet** dataset | Fei-Fei Li et al., Stanford | 2009 | 1.2M images, 1,000 categories |
| **EfficientNet** architecture | Mingxing Tan & Quoc V. Le, **Google Brain** | 2019 | Scales width/depth/resolution jointly |
| **Pretrained weights** | Ross Wightman → **Hugging Face** (`timm`) | ongoing | `timm.create_model("efficientnet_b3", pretrained=True)` downloads these |

EfficientNet was designed for natural images (cats, dogs, cars), but transfers well to spectrograms because texture and edge patterns that indicate bird calls share structural similarity with natural image features.

#### AudioSet pretrained — CNN14 (PANNs)

| Component | Creator | Year | Notes |
|---|---|---|---|
| **AudioSet** dataset | **Google Research** | 2017 | 2M YouTube clips, 527 sound categories |
| **CNN14 / PANNs** architecture | Qiuqiang Kong et al., **University of Surrey** | 2020 | Paper: *"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"* |
| **Pretrained weights** | University of Surrey — hosted on **Zenodo** | 2020 | Open-access; downloaded via `load_pretrained_panns()` in `model.py` |

PANNs were trained directly on sound — AudioSet includes bird calls, rain, wind, and animal vocalizations. This makes CNN14's priors much closer to the Pantanal task than ImageNet-trained models, despite having a simpler architecture.

> **Why this matters for the competition:** EfficientNet is faster to fine-tune and benefits from a larger pretrained model. CNN14 starts with stronger audio priors. Ensembling both exploits complementary strengths.

### Why EfficientNet on spectrograms?

Converting audio to log-mel spectrograms turns a 1D signal classification problem into a 2D image classification problem. EfficientNet-B3 pretrained on ImageNet provides a strong feature extractor even for spectrograms — texture patterns that indicate bird calls share structural similarity with natural image textures.

### Why Focal Loss?

Standard BCE loss averages over all species × samples. With 234 species and heavy class imbalance, the loss is dominated by easy true-negative predictions. Focal loss applies a modulating factor `(1 - p_t)^γ` that down-weights well-classified examples, forcing the model to focus on difficult, rare species.

### Why train on soundscape labels too?

Test-time data is continuous fieldwork recordings — ambient noise, distant calls, multiple species simultaneously. Short clean clips from Xeno-canto/iNaturalist have different acoustic properties. The labeled soundscape windows close this gap without requiring any extra data collection.

---

## How to Run

### Install dependencies

```bash
cd pantanal-audio-classification
pip install -r requirements.txt
```

### Train all 5 folds

```bash
# Single fold (fold 0)
python3 src/train.py --config configs/default.json --fold 0

# All folds in parallel (if you have multiple GPUs)
for i in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$i python src/train.py --fold $i &
done
wait
```

### Generate submission

```bash
python src/inference.py \
  --config configs/default.json \
  --data_dir ../birdclef-2026-2 \
  --checkpoint_dir checkpoints \
  --output_csv submission.csv
```

### Configuration

All hyperparameters live in [`configs/default.json`](configs/default.json):

```json
{
  "model":         "efficientnet_b3",   // or "cnn14"
  "epochs":        30,
  "batch_size":    32,
  "lr":            1e-3,
  "focal_gamma":   2.0,                 // higher = more focus on rare species
  "mixup_alpha":   0.4,                 // 0 = disable mixup
  "noise_files":   []                   // paths to background noise .ogg files
}
```

---

## Improvement Roadmap

| Priority | Technique | Expected Gain |
|---|---|---|
| High | Add Xeno-canto clips for rare species (<10 clips) | +AUC on tail classes |
| High | Filter train.csv by `rating >= 3.5` | Cleaner signal |
| Medium | PANNs CNN14 pretrained weights | Stronger audio priors vs ImageNet |
| Medium | ConvNeXt-Small backbone | Often beats EfficientNet on spectrograms |
| Medium | Pseudo-labeling on unlabeled soundscapes | More training data |
| Low | Per-location prior (taxonomy × GPS coordinates) | Site-specific species bias |
| Low | Multi-scale spectrograms (64 + 128 + 256 mel bins) | Richer frequency resolution |

---

## Recording Locations

All data collected at **Pantanal, Mato Grosso do Sul, Brazil**

```
Latitude:  -16.5 to -21.6
Longitude: -55.9 to -57.6
```

One of the world's largest tropical wetlands; 650+ bird species; threatened by agricultural
expansion, wildfires, and seasonal flooding.
