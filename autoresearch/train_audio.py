"""
BirdCLEF 2026 — Autoresearch training script.
Adapted from Karpathy's autoresearch for audio classification on MPS/CPU/CUDA.

The agent edits this file. The metric is val_auc (higher is better).
Training runs for TIME_BUDGET seconds (wall clock, excluding startup).

Usage: python3 train_audio.py
Output (last lines, parsed by agent):
  val_auc:          0.XXXXXX
  training_seconds: XXX.X
  total_seconds:    XXX.X
"""

import sys
import os
import time
import random
import math
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

from dataset import ClipDataset, SoundscapeTrainDataset, mixup_collate, build_species_index
from model import EfficientNetClassifier, CNN14Classifier

# ---------------------------------------------------------------------------
# Hyperparameters — agent edits these
# ---------------------------------------------------------------------------

# Model
MODEL        = "efficientnet_b3"  # "efficientnet_b3" or "cnn14"
DROP_RATE    = 0.3

# Optimization
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2

# Loss
FOCAL_GAMMA  = 2.0
POS_WEIGHT_CLAMP = (1.0, 10.0)

# Augmentation
MIXUP_ALPHA  = 0.4   # 0 = disable mixup
USE_SPEC_AUG = True  # enable SpecAugment in dataset

# Training
BATCH_SIZE   = 32
TIME_BUDGET  = 900   # seconds of actual training (15 min)
FOLD         = 0     # which fold to validate on

# ---------------------------------------------------------------------------
# Fixed constants — do not modify
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "birdclef-2026-2"
SEED     = 42

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"Device: {device}")

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from functools import partial

species_list, species_to_idx = build_species_index(DATA_DIR / "taxonomy.csv")
num_classes = len(species_list)

train_df = pd.read_csv(DATA_DIR / "train.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_df["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
    train_df.loc[val_idx, "fold"] = fold_idx

trn_df = train_df[train_df["fold"] != FOLD].reset_index(drop=True)
val_df = train_df[train_df["fold"] == FOLD].reset_index(drop=True)

sl_df = pd.read_csv(DATA_DIR / "train_soundscapes_labels.csv")

clip_ds = ClipDataset(trn_df, species_to_idx, num_classes,
                      audio_dir=DATA_DIR / "train_audio", augment=USE_SPEC_AUG)
soundscape_ds = SoundscapeTrainDataset(sl_df, species_to_idx, num_classes,
                                       soundscape_dir=DATA_DIR / "train_soundscapes",
                                       augment=USE_SPEC_AUG)
val_ds = ClipDataset(val_df, species_to_idx, num_classes,
                     audio_dir=DATA_DIR / "train_audio", augment=False)

collate = partial(mixup_collate, alpha=MIXUP_ALPHA) if MIXUP_ALPHA > 0 else None
pin = device.type == "cuda"
nw = 2

train_loader = DataLoader(
    ConcatDataset([clip_ds, soundscape_ds]),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=nw, pin_memory=pin,
    collate_fn=collate, drop_last=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
    num_workers=nw, pin_memory=pin,
)

# Model
if MODEL == "efficientnet_b3":
    model = EfficientNetClassifier(num_classes, pretrained=True, drop_rate=DROP_RATE)
else:
    model = CNN14Classifier(num_classes, drop_rate=DROP_RATE)
model = model.to(device)

# Positive class weights
counts = np.zeros(num_classes)
for label in trn_df["primary_label"].astype(str):
    if label in species_to_idx:
        counts[species_to_idx[label]] += 1
pos_weight = torch.tensor(
    (len(trn_df) - counts) / (counts + 1), dtype=torch.float32
).clamp(*POS_WEIGHT_CLAMP).to(device)

# Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
        return ((1 - pt) ** self.gamma * bce).mean()

loss_fn = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# LR schedule
def cosine_warmup_schedule(optimizer, warmup_steps, total_steps):
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

# Estimate steps for warmup (we'll update as we go)
steps_per_epoch = len(train_loader)
warmup_steps = steps_per_epoch * WARMUP_EPOCHS
# Use a large total_steps so schedule doesn't finish too early
total_steps = steps_per_epoch * 100
scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

use_amp = device.type == "cuda"
scaler = torch.GradScaler() if use_amp else None

# ---------------------------------------------------------------------------
# Training loop — runs until TIME_BUDGET seconds elapsed
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
best_val_auc = 0.0

print(f"Time budget: {TIME_BUDGET}s | Steps/epoch: {steps_per_epoch}")

while True:
    model.train()
    epoch += 1
    for batch in train_loader:
        t0 = time.time()
        specs = batch["spectrogram"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = loss_fn(model(specs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(specs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        t1 = time.time()
        total_training_time += t1 - t0
        step += 1

        if step % 50 == 0:
            pct = 100 * total_training_time / TIME_BUDGET
            print(f"\repoch {epoch} step {step} ({pct:.1f}%) loss={loss.item():.4f}   ",
                  end="", flush=True)

        if total_training_time >= TIME_BUDGET:
            break

    if total_training_time >= TIME_BUDGET:
        break

    # Validate at end of each epoch
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            specs = batch["spectrogram"].to(device)
            labels = batch["labels"].to(device)
            preds = torch.sigmoid(model(specs)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    aucs = []
    for col in range(targets.shape[1]):
        if targets[:, col].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, col], preds[:, col]))
            except Exception:
                pass
    val_auc = float(np.mean(aucs)) if aucs else 0.0
    best_val_auc = max(best_val_auc, val_auc)
    print(f"\nepoch {epoch} | val_auc={val_auc:.4f} | best={best_val_auc:.4f}")

print()

# Final validation
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_loader:
        specs = batch["spectrogram"].to(device)
        labels = batch["labels"].to(device)
        preds = torch.sigmoid(model(specs)).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())
preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)
aucs = []
for col in range(targets.shape[1]):
    if targets[:, col].sum() > 0:
        try:
            aucs.append(roc_auc_score(targets[:, col], preds[:, col]))
        except Exception:
            pass
final_val_auc = float(np.mean(aucs)) if aucs else 0.0

t_end = time.time()
print("---")
print(f"val_auc:          {final_val_auc:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"epochs_completed: {epoch}")
print(f"steps_completed:  {step}")
