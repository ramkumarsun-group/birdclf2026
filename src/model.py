"""
Model architectures for Pantanal audio classification.

Two options:
  1. EfficientNetClassifier — fast, proven BirdCLEF baseline
  2. PANNsClassifier       — uses CNN14 pretrained on AudioSet (stronger priors)

Both share the same output contract: logits of shape (B, num_classes).
"""
import torch
import torch.nn as nn
import timm


# ── Option 1: EfficientNet on mel spectrograms ─────────────────────────────

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned on log-mel spectrograms.

    Input:  (B, 1, 128, T) — single-channel spectrogram
    Output: (B, num_classes) logits (sigmoid applied externally during loss)
    """

    def __init__(self, num_classes: int, model_name: str = "efficientnet_b3",
                 pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()
        # Convert 1-channel spec to 3-channel by repeating (ImageNet expects RGB)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,        # timm handles 1→3 weight averaging internally
            num_classes=0,     # remove classifier head
            global_pool="avg",
            drop_rate=drop_rate,
        )
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, feature_dim)
        return self.head(features)    # (B, num_classes)


# ── Option 2: PANNs CNN14 (stronger audio priors) ─────────────────────────

class CNN14Block(nn.Module):
    """Single conv block used in CNN14 architecture."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class CNN14Classifier(nn.Module):
    """
    Lightweight CNN14-style model.
    Load pretrained PANNs weights with load_pretrained_panns() below.

    Input:  (B, 1, 128, T)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int, drop_rate: float = 0.3):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.layers = nn.Sequential(
            CNN14Block(1, 64),
            CNN14Block(64, 128),
            CNN14Block(128, 256),
            CNN14Block(256, 512),
            CNN14Block(512, 1024),
            CNN14Block(1024, 2048),
        )
        self.dropout = nn.Dropout(drop_rate)
        # After 6 pooling ops on (128, T): ~(2, T//64)
        self.head = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn0(x)
        x = self.layers(x)                 # (B, 2048, H', W')
        x = x.mean(dim=[2, 3])             # global avg pool
        x = self.dropout(x)
        return self.head(x)


def load_pretrained_panns(model: CNN14Classifier, checkpoint_path: str) -> CNN14Classifier:
    """
    Load weights from official PANNs CNN14 checkpoint.
    Skips final classifier layer (class count differs).
    Download from: https://zenodo.org/record/3960586
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    pretrained = state.get("model", state)
    own_state = model.state_dict()

    loaded, skipped = 0, 0
    for name, param in pretrained.items():
        if name in own_state and own_state[name].shape == param.shape:
            own_state[name].copy_(param)
            loaded += 1
        else:
            skipped += 1

    print(f"PANNs weights loaded: {loaded} matched, {skipped} skipped")
    return model


# ── Ensemble wrapper ───────────────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """
    Averages sigmoid outputs from multiple trained models.
    Used at inference time — not trained end-to-end.
    """

    def __init__(self, models: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = sum(
            w * torch.sigmoid(m(x))
            for m, w in zip(self.models, self.weights)
        )
        return probs  # already probability, not logits
