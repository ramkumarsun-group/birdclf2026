# BirdCLEF 2026 — Autoresearch Program

## Goal
Maximize `val_auc` (ROC-AUC, higher is better) for audio classification of 234 wildlife species
from the Pantanal wetlands. The model runs on Apple MPS (Mac GPU).

## Baseline
- Model: EfficientNet-B3
- val_auc: ~0.974 (fold 0, 30 epochs full training)
- Each autoresearch experiment: 15-minute time budget

## The file you edit
`train_audio.py` — contains all hyperparameters and the training loop.
Only edit the **Hyperparameters** section (clearly marked). Do not modify the Fixed Constants section.

## How to run an experiment
```bash
cd /Users/ramkumarsundaram/Documents/pantanal-audio-classification
python3 autoresearch/train_audio.py 2>&1 | tail -10
```

Parse the output lines:
- `val_auc: X.XXXXXX` — the metric (higher is better)
- `training_seconds: XXX` — actual training time

## Experiment protocol
1. Read the current `train_audio.py` to understand the baseline
2. Make ONE change at a time (one hyperparameter or one small code modification)
3. Run the experiment and record `val_auc`
4. If val_auc improved → keep the change (update train_audio.py permanently)
5. If val_auc did not improve → revert the change
6. Log every experiment in this file under ## Experiment Log

## Research directions to explore (in rough priority order)

### 1. Augmentation strength
- Try MIXUP_ALPHA: 0.2, 0.3, 0.5, 0.6
- Try USE_SPEC_AUG: True vs False
- Try adding time-stretch or pitch-shift to dataset augmentation

### 2. Loss function tuning
- Try FOCAL_GAMMA: 1.0, 1.5, 2.5, 3.0
- Try POS_WEIGHT_CLAMP upper bound: 5.0, 8.0, 15.0

### 3. Learning rate & schedule
- Try LR: 5e-4, 2e-3, 3e-3
- Try WEIGHT_DECAY: 1e-3, 1e-5
- Try WARMUP_EPOCHS: 1, 3, 5

### 4. Model architecture
- Try DROP_RATE: 0.1, 0.2, 0.4, 0.5
- Try MODEL: "cnn14" (different inductive bias)
- Try changing the head: add BatchNorm between Linear layers

### 5. Batch size
- Try BATCH_SIZE: 16, 64 (may need to adjust LR proportionally)

### 6. Training loop improvements
- Try label smoothing (add eps=0.05 to targets)
- Try gradient clipping threshold: 0.5, 2.0
- Try cosine annealing with restarts instead of warmup+decay

## Experiment Log

| # | Change | val_auc | Delta | Kept? |
|---|--------|---------|-------|-------|
| 0 | Baseline (MIXUP_ALPHA=0.4, FOCAL_GAMMA=2.0, LR=1e-3) | TBD | — | — |

## Notes
- MPS does not support bfloat16 well — keep float32 (default)
- `torch.compile` is not stable on MPS — do not add it
- The dataset loads from `birdclef-2026-2/` relative to project root
- Validation uses fold 0 consistently across all experiments for fair comparison
