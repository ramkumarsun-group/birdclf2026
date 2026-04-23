"""
Dataset for BirdCLEF 2026 — Pantanal wildlife audio classification.

Key facts from the competition data:
  - 234 target species (mix of eBird codes, iNat IDs, insect sonotypes 47158son01-25)
  - train.csv filename column = "{primary_label}/{file}.ogg" (relative to train_audio/)
  - train_soundscapes_labels.csv has 5-second labeled windows — use as additional training data
  - Submission: row_id = "{soundscape_stem}_{end_second}", windows at 5, 10, 15...
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa


# ── Constants ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 32_000
WINDOW_SECS = 5
WINDOW_LEN = SAMPLE_RATE * WINDOW_SECS
N_FFT = 1024
HOP_LENGTH = 320          # ~10ms hop → ~500 time frames per 5s
N_MELS = 128
FMIN = 50
FMAX = 14_000


# ── Audio helpers ──────────────────────────────────────────────────────────

def load_clip(path: str | Path, start_sec: float = 0.0) -> np.ndarray:
    """Load exactly 5 seconds starting at start_sec; zero-pads if too short."""
    audio, _ = librosa.load(
        str(path), sr=SAMPLE_RATE, mono=True,
        offset=start_sec, duration=WINDOW_SECS,
    )
    if len(audio) < WINDOW_LEN:
        audio = np.pad(audio, (0, WINDOW_LEN - len(audio)))
    return audio[:WINDOW_LEN]


def audio_to_melspec(audio: np.ndarray) -> np.ndarray:
    """Waveform → normalized log-mel spectrogram, shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    return log_mel.astype(np.float32)


# ── Augmentations ──────────────────────────────────────────────────────────

def spec_augment(spec: np.ndarray,
                 num_freq: int = 2, freq_size: int = 15,
                 num_time: int = 2, time_size: int = 30) -> np.ndarray:
    """Zero out random frequency and time bands (SpecAugment)."""
    spec = spec.copy()
    n_mels, n_time = spec.shape
    for _ in range(num_freq):
        f0 = random.randint(0, n_mels - freq_size)
        spec[f0:f0 + freq_size, :] = 0.0
    for _ in range(num_time):
        t0 = random.randint(0, n_time - time_size)
        spec[:, t0:t0 + time_size] = 0.0
    return spec


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Mix background noise at given SNR."""
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    noise = noise[:len(audio)]
    scale = np.sqrt(
        (np.mean(audio ** 2) + 1e-9) /
        (np.mean(noise ** 2) + 1e-9) /
        (10 ** (snr_db / 10))
    )
    return (audio + scale * noise).clip(-1.0, 1.0)


# ── Label helpers ──────────────────────────────────────────────────────────

def build_species_index(taxonomy_csv: str | Path) -> tuple[list[str], dict[str, int]]:
    """
    Load ordered species list from taxonomy.csv.
    Returns (species_list, species_to_idx) using primary_label column.
    Order matches the submission CSV column order.
    """
    df = pd.read_csv(taxonomy_csv)
    species_list = df["primary_label"].astype(str).tolist()
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    return species_list, species_to_idx


def parse_label_field(value) -> list[str]:
    """Parse secondary_labels field which may be a Python-style list string, e.g. "['abc', 'def']"."""
    if not isinstance(value, str) or value in ("[]", "", "nan"):
        return []
    value = value.strip("[]").replace("'", "").replace('"', "")
    return [v.strip() for v in value.split(",") if v.strip()]


# ── Training datasets ──────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    Short clip dataset from train.csv.
    train.csv filename column = "{primary_label}/{file}.ogg" relative to train_audio/.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        species_to_idx: dict[str, int],
        num_classes: int,
        audio_dir: str | Path,
        augment: bool = True,
        noise_files: list[str] | None = None,
        secondary_weight: float = 0.4,
    ):
        self.df = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.audio_dir = Path(audio_dir)
        self.augment = augment
        self.noise_files = noise_files or []
        self.secondary_weight = secondary_weight

    def __len__(self) -> int:
        return len(self.df)

    def _make_label(self, row: pd.Series) -> torch.Tensor:
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        primary = str(row["primary_label"])
        if primary in self.species_to_idx:
            label[self.species_to_idx[primary]] = 1.0
        for s in parse_label_field(row.get("secondary_labels", "[]")):
            if s in self.species_to_idx:
                label[self.species_to_idx[s]] = max(
                    label[self.species_to_idx[s]], self.secondary_weight
                )
        return label

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.audio_dir / str(row["filename"])

        # Random crop within the file
        try:
            duration = librosa.get_duration(path=str(path))
            max_start = max(0.0, duration - WINDOW_SECS)
            start = random.uniform(0, max_start) if self.augment else 0.0
        except Exception:
            start = 0.0

        audio = load_clip(path, start_sec=start)

        if self.augment and self.noise_files and random.random() < 0.3:
            noise_path = random.choice(self.noise_files)
            try:
                noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
                audio = add_noise(audio, noise, snr_db=random.uniform(5, 20))
            except Exception:
                pass

        spec = audio_to_melspec(audio)
        if self.augment:
            spec = spec_augment(spec)

        return {
            "spectrogram": torch.from_numpy(spec).unsqueeze(0),
            "labels": self._make_label(row),
        }


class SoundscapeTrainDataset(Dataset):
    """
    Uses train_soundscapes_labels.csv as labeled training data.
    Each row is a 5-second window with ≥1 species labels.
    This is the same distribution as the test set — highly valuable.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        species_to_idx: dict[str, int],
        num_classes: int,
        soundscape_dir: str | Path,
        augment: bool = True,
    ):
        self.df = labels_df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.num_classes = num_classes
        self.soundscape_dir = Path(soundscape_dir)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _parse_time(t: str) -> float:
        """'HH:MM:SS' → seconds."""
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    def _make_label(self, label_str: str) -> torch.Tensor:
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for s in str(label_str).split(";"):
            s = s.strip()
            if s in self.species_to_idx:
                label[self.species_to_idx[s]] = 1.0
        return label

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.soundscape_dir / row["filename"]
        start_sec = self._parse_time(row["start"])

        audio = load_clip(path, start_sec=start_sec)
        spec = audio_to_melspec(audio)
        if self.augment:
            spec = spec_augment(spec)

        return {
            "spectrogram": torch.from_numpy(spec).unsqueeze(0),
            "labels": self._make_label(row["primary_label"]),
        }


# ── Mixup collate ──────────────────────────────────────────────────────────

def mixup_collate(batch: list[dict], alpha: float = 0.4) -> dict:
    specs = torch.stack([b["spectrogram"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    if alpha > 0 and random.random() < 0.5:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(len(batch))
        specs = lam * specs + (1 - lam) * specs[perm]
        labels = lam * labels + (1 - lam) * labels[perm]
    return {"spectrogram": specs, "labels": labels}


# ── Inference: sliding window over soundscape ──────────────────────────────

class SoundscapeInferDataset(Dataset):
    """
    Slides a non-overlapping 5-second window over a soundscape for inference.
    row_id = "{stem}_{end_second}" — matches submission format exactly.
    """

    def __init__(self, filepath: str | Path):
        path = Path(filepath)
        self.stem = path.stem
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)

        self.windows: list[tuple[np.ndarray, int]] = []
        for start in range(0, len(audio), WINDOW_LEN):
            chunk = audio[start:start + WINDOW_LEN]
            if len(chunk) < WINDOW_LEN:
                chunk = np.pad(chunk, (0, WINDOW_LEN - len(chunk)))
            end_sec = (start // WINDOW_LEN + 1) * WINDOW_SECS
            self.windows.append((chunk, end_sec))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        audio, end_sec = self.windows[idx]
        spec = audio_to_melspec(audio)
        return {
            "spectrogram": torch.from_numpy(spec).unsqueeze(0),
            "row_id": f"{self.stem}_{end_sec}",
        }
