"""
Inference for BirdCLEF 2026.

Submission format:
  row_id = "{soundscape_stem}_{end_second}"   (non-overlapping 5s windows)
  234 species columns with probability scores.

Example:
  BC2026_Test_0001_S05_20250227_010002_5   → seconds 0–5
  BC2026_Test_0001_S05_20250227_010002_10  → seconds 5–10
"""
import json
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SoundscapeInferDataset, build_species_index, audio_to_melspec, spec_augment
from model import EfficientNetClassifier, CNN14Classifier, EnsembleModel


# ── Model loading ──────────────────────────────────────────────────────────

def load_single_model(cfg: dict, num_classes: int, ckpt_path: str,
                      device: torch.device) -> torch.nn.Module:
    if cfg["model"] == "efficientnet_b3":
        model = EfficientNetClassifier(num_classes, pretrained=False)
    else:
        model = CNN14Classifier(num_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.eval().to(device)


def build_ensemble(cfg: dict, num_classes: int, ckpt_dir: str,
                   device: torch.device) -> EnsembleModel:
    models = []
    for fold in range(cfg["num_folds"]):
        ckpt = Path(ckpt_dir) / f"best_fold{fold}.pt"
        if ckpt.exists():
            models.append(load_single_model(cfg, num_classes, str(ckpt), device))
            print(f"  Loaded fold {fold}")
    print(f"Ensemble: {len(models)} models")
    return EnsembleModel(models).to(device)


# ── TTA ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_tta(model: torch.nn.Module, specs: torch.Tensor,
                     device: torch.device, n_aug: int = 4) -> np.ndarray:
    """Average predictions over original + n_aug-1 SpecAugmented versions."""
    probs_list = []

    if isinstance(model, EnsembleModel):
        probs_list.append(model(specs.to(device)).cpu().numpy())
    else:
        probs_list.append(torch.sigmoid(model(specs.to(device))).cpu().numpy())

    for _ in range(n_aug - 1):
        aug = torch.stack([
            torch.from_numpy(spec_augment(s.squeeze(0).numpy())).unsqueeze(0)
            for s in specs
        ])
        if isinstance(model, EnsembleModel):
            probs_list.append(model(aug.to(device)).cpu().numpy())
        else:
            probs_list.append(torch.sigmoid(model(aug.to(device))).cpu().numpy())

    return np.mean(probs_list, axis=0)


# ── Soundscape prediction ──────────────────────────────────────────────────

@torch.no_grad()
def predict_soundscape(model: torch.nn.Module, filepath: str | Path,
                       device: torch.device, batch_size: int = 32,
                       use_tta: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [row_id, species1, species2, ...].
    One row per 5-second window.
    """
    ds = SoundscapeInferDataset(filepath)
    if len(ds) == 0:
        return pd.DataFrame()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_row_ids, all_probs = [], []

    for batch in loader:
        specs = batch["spectrogram"]
        row_ids = batch["row_id"]
        if use_tta:
            probs = predict_with_tta(model, specs, device)
        elif isinstance(model, EnsembleModel):
            probs = model(specs.to(device)).cpu().numpy()
        else:
            probs = torch.sigmoid(model(specs.to(device))).cpu().numpy()
        all_row_ids.extend(row_ids)
        all_probs.append(probs)

    return np.concatenate(all_probs), all_row_ids


def generate_submission(
    model: torch.nn.Module,
    test_dir: str | Path,
    species_list: list[str],
    device: torch.device,
    output_csv: str,
    batch_size: int = 32,
    use_tta: bool = True,
):
    """Process all soundscapes and write submission CSV."""
    test_dir = Path(test_dir)
    audio_exts = {".ogg", ".flac", ".wav", ".mp3"}
    files = [f for f in test_dir.rglob("*") if f.suffix in audio_exts]
    print(f"Test soundscapes: {len(files)}")

    all_row_ids, all_probs = [], []
    for fp in tqdm(files, desc="Inference"):
        probs, row_ids = predict_soundscape(model, fp, device, batch_size, use_tta)
        all_row_ids.extend(row_ids)
        all_probs.append(probs)

    probs_matrix = np.concatenate(all_probs)  # (N_windows, 234)
    df = pd.DataFrame(probs_matrix, columns=species_list)
    df.insert(0, "row_id", all_row_ids)
    df.to_csv(output_csv, index=False)
    print(f"Submission saved: {output_csv}  ({len(df)} rows × {len(species_list)} species)")
    return df


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data_dir", required=True,
                        help="Path to birdclef-2026-2/ folder")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_csv", default="submission.csv")
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    data_dir = Path(args.data_dir)
    species_list, _ = build_species_index(data_dir / "taxonomy.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_ensemble(cfg, len(species_list), args.checkpoint_dir, device)
    generate_submission(
        model,
        test_dir=data_dir / "test_soundscapes",
        species_list=species_list,
        device=device,
        output_csv=args.output_csv,
        use_tta=not args.no_tta,
    )
