"""
Training loop for BirdCLEF 2026 — Pantanal audio classification.

Data sources used:
  1. train.csv + train_audio/  — 35,549 labeled short clips
  2. train_soundscapes_labels.csv + train_soundscapes/ — labeled soundscape windows

Species list loaded from taxonomy.csv (234 species, ordered to match submission columns).
"""
import argparse
import json
import random
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dataset import (
    ClipDataset, SoundscapeTrainDataset, mixup_collate, build_species_index
)
from model import EfficientNetClassifier, CNN14Classifier


# ── Loss ───────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss for multi-label, with per-class positive weighting."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        pt = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
        return ((1 - pt) ** self.gamma * bce).mean()


# ── LR schedule ────────────────────────────────────────────────────────────

def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ── Metrics ────────────────────────────────────────────────────────────────

def mean_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Mean per-species ROC-AUC, skipping species with no positives."""
    aucs = []
    for col in range(targets.shape[1]):
        if targets[:, col].sum() > 0:
            try:
                aucs.append(roc_auc_score(targets[:, col], preds[:, col]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, scaler, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"
    bar = tqdm(loader, desc=f"Epoch {epoch:03d}/{num_epochs} [train]", unit="batch", leave=False)
    for batch in bar:
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
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    bar = tqdm(loader, desc="              [ val ]", unit="batch", leave=False)
    for batch in bar:
        specs = batch["spectrogram"].to(device)
        labels = batch["labels"].to(device)
        logits = model(specs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return total_loss / len(loader), mean_auc(targets, preds)


# ── Main ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg: dict):
    seed_everything(cfg["seed"])
    device = get_device()
    data_dir = Path(cfg["data_dir"])

    # ── Species from taxonomy (preserves submission column order) ──
    species_list, species_to_idx = build_species_index(data_dir / "taxonomy.csv")
    num_classes = len(species_list)
    print(f"Species: {num_classes} | Device: {device}")

    # ── Load train.csv ──
    train_df = pd.read_csv(data_dir / "train.csv")

    # ── 5-fold CV stratified by primary_label ──
    skf = StratifiedKFold(n_splits=cfg["num_folds"], shuffle=True, random_state=cfg["seed"])
    train_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
        train_df.loc[val_idx, "fold"] = fold_idx

    fold = cfg["fold"]
    trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
    print(f"Fold {fold}: train={len(trn_df)}, val={len(val_df)}")

    # ── Load soundscape labels (bonus labeled data on the target distribution) ──
    sl_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    print(f"Soundscape training windows: {len(sl_df)}")

    # ── Datasets ──
    clip_ds = ClipDataset(
        trn_df, species_to_idx, num_classes,
        audio_dir=data_dir / "train_audio",
        augment=True,
        noise_files=cfg.get("noise_files", []),
    )
    soundscape_ds = SoundscapeTrainDataset(
        sl_df, species_to_idx, num_classes,
        soundscape_dir=data_dir / "train_soundscapes",
        augment=True,
    )
    val_ds = ClipDataset(
        val_df, species_to_idx, num_classes,
        audio_dir=data_dir / "train_audio",
        augment=False,
    )

    collate = partial(mixup_collate, alpha=cfg["mixup_alpha"])
    pin = device.type == "cuda"   # pin_memory only works on CUDA
    nw = cfg["num_workers"]
    train_loader = DataLoader(
        ConcatDataset([clip_ds, soundscape_ds]),
        batch_size=cfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pin,
        collate_fn=collate, drop_last=True,
        persistent_workers=nw > 0,  # keep workers alive between epochs — prevents "too many open files"
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0,
    )

    # ── Model ──
    if cfg["model"] == "efficientnet_b3":
        model = EfficientNetClassifier(num_classes, pretrained=True)
    elif cfg["model"] == "cnn14":
        model = CNN14Classifier(num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")
    model = model.to(device)

    # ── Positive class weights (from clip training set) ──
    counts = np.zeros(num_classes)
    for label in trn_df["primary_label"].astype(str):
        if label in species_to_idx:
            counts[species_to_idx[label]] += 1
    pos_weight = torch.tensor(
        (len(trn_df) - counts) / (counts + 1), dtype=torch.float32
    ).clamp(1.0, 10.0).to(device)

    loss_fn = FocalLoss(gamma=cfg["focal_gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = len(train_loader) * cfg["warmup_epochs"]
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.GradScaler() if device.type == "cuda" else None

    # ── Resume from checkpoint if available ──
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_fold{fold}.pt"
    resume_path = out_dir / f"resume_fold{fold}.pt"

    start_epoch = 1
    best_auc = 0.0

    if cfg.get("resume"):
        if resume_path.exists():
            # Full resume — model + optimizer + scheduler
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_auc = checkpoint["best_auc"]
            print(f"Resuming from epoch {start_epoch} (best auc so far: {best_auc:.4f})")
        elif ckpt_path.exists():
            # Partial resume — model weights only (optimizer/scheduler reset)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            best_auc = checkpoint["auc"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Partial resume from best checkpoint (epoch {checkpoint['epoch']}, auc={best_auc:.4f})")
            print("  Note: optimizer and scheduler reset — training will re-warm-up briefly")
        else:
            print("No checkpoint found — starting from scratch")

    # ── Loop ──
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        trn_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, scaler, epoch, cfg["epochs"])
        val_loss, val_auc = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | trn={trn_loss:.4f} | val={val_loss:.4f} | auc={val_auc:.4f}")

        # Save best model weights
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "auc": val_auc, "species_list": species_list}, ckpt_path)
            print(f"  ✓ Best checkpoint saved (auc={val_auc:.4f})")

        # Save full resume state after every epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
        }, resume_path)

    print(f"\nFold {fold} best AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from last saved epoch")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    cfg["fold"] = args.fold
    cfg["resume"] = args.resume
    main(cfg)
