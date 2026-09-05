"""B2: a mel-spectrogram CNN, built the way the literature builds them.

**What changed in A7.2, and why.** The first version of this file sized its
channel widths so the parameter count landed within a few percent of the Task 2
GNN, on the theory that matching capacity made the comparison fair. That was a
mistake, and an expensive one: it scored 0.1654 macro-F1 on MTAT top-50 where
published mel-CNNs reach roughly 0.38-0.41, so the "result" it produced was a
statement about a crippled baseline rather than about inductive bias.

Three things were wrong, and all three are fixed here:

* **Parameter matching was the wrong constraint.** A GNN over 32 nodes of
  96-dimensional pooled features and a CNN over a 128 x 129 spectrogram do not
  become comparable by having the same parameter count -- convolutional weights
  are reused across every time-frequency position, so the same count buys wildly
  different amounts of computation. Equalising the *compute budget* (same GPU,
  same epoch cap, same early-stopping rule) and reporting parameters and
  wall-clock alongside is the honest framing, so that is what the results tables
  now carry.
* **The input was pooled to death.** B2 read a cache mean-pooled to 256 columns
  per track, 8.8 frames/second, so a 3-second window was 26 columns wide. It now
  reads the native-resolution cache at 43.07 frames/second.
* **It saw whole tracks.** One global descriptor per 29-second clip was being
  asked to explain 50 local tags. Training now happens on 3-second excerpts and
  inference averages per-chunk probabilities, which is the standard protocol.

The architecture is the "short-chunk CNN" that is the usual strong MTAT
baseline: seven 3x3 convolution blocks with 2x2 max-pooling, doubling widths,
about 3.4M parameters. At 128 mels x 129 frames the seven pools take the map to
1x1, so the head sees a single vector per chunk.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_parameters, get_logger

LOGGER = get_logger("gbmc.cnn")

__all__ = ["MelCNN", "ConvBlock", "SHORT_CHUNK_CHANNELS"]

#: widths of the seven blocks; the last one feeds the classifier head
SHORT_CHUNK_CHANNELS = (64, 128, 128, 256, 256, 256, 512)


class ConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> MaxPool2x2, the short-chunk CNN's unit."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool) if pool != (1, 1) else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class MelCNN(nn.Module):
    """Short-chunk mel CNN for multi-label tagging and/or single-label genre.

    ``forward`` accepts either a plain batch of chunks ``[B, 1, n_mels, F]`` or a
    batch of chunk *stacks* ``[B, C, 1, n_mels, F]``. In the second case every
    chunk is scored independently and the per-chunk **probabilities** are
    averaged into one prediction per clip -- averaging probabilities rather than
    logits is what the MTAT protocol specifies, and it matters: a logit average
    lets one confident chunk dominate the whole clip.

    ``n_tags=0`` builds a genre-only model, so the reported parameter count
    describes the model actually being trained.
    """

    def __init__(self, n_tags: int, n_mels: int = 128, channels=SHORT_CHUNK_CHANNELS,
                 dropout: float = 0.5, in_channels: int = 1, n_genres: int = 0,
                 target_params: int | None = None):
        super().__init__()
        if target_params is not None:
            raise ValueError(
                "target_params was removed in A7.2: parameter-matching B2 to the "
                "GNN is what produced the implausible 0.165 macro-F1. Compare on "
                "compute budget and report both counts instead."
            )
        if not int(n_tags) and not int(n_genres):
            raise ValueError("MelCNN needs n_tags > 0 or n_genres > 0")

        self.n_tags = int(n_tags)
        self.n_genres = int(n_genres)
        self.n_mels = int(n_mels)
        self.in_channels = int(in_channels)
        self.dropout = float(dropout)
        self.channels = tuple(int(c) for c in channels)

        widths = [self.in_channels, *self.channels]
        blocks = []
        freq = self.n_mels
        for i in range(len(self.channels)):
            # stop halving the frequency axis once it would vanish
            pool_f = 2 if freq >= 2 else 1
            freq = max(freq // pool_f, 1)
            blocks.append(ConvBlock(widths[i], widths[i + 1], pool=(pool_f, 2)))
        self.blocks = nn.Sequential(*blocks)

        d = self.channels[-1] * 2                     # mean || max over what is left
        self.drop = nn.Dropout(self.dropout)
        self.tag_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(d // 2, self.n_tags),
        ) if self.n_tags else None
        self.genre_head = nn.Linear(d, self.n_genres) if self.n_genres else None

        LOGGER.info("MelCNN: %d blocks %s, %d mels -> %s trainable params",
                    len(self.channels), list(self.channels), self.n_mels,
                    f"{count_parameters(self):,}")

    # -- forward ---------------------------------------------------------- #
    def _encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)                        # [B, n_mels, F] -> [B,1,n_mels,F]
        h = self.blocks(x)
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1)

    def _heads(self, pooled):
        out = {"z": pooled}
        if self.tag_head is not None:
            out["tag_logits"] = self.tag_head(self.drop(pooled))
        if self.genre_head is not None:
            out["genre_logits"] = self.genre_head(self.drop(pooled))
        return out

    def forward(self, x, return_features: bool = False):
        if x.dim() == 5:
            # [B, C, 1, n_mels, F]: score every chunk, then average probabilities
            b, c = x.shape[0], x.shape[1]
            flat = self._heads(self._encode(x.reshape(b * c, *x.shape[2:])))
            out = {"z": flat["z"].view(b, c, -1).mean(dim=1)}
            if "tag_logits" in flat:
                probs = torch.sigmoid(flat["tag_logits"].float()).view(b, c, -1).mean(dim=1)
                out["tag_probs"] = probs
                # logits back out for the loss, from the averaged probability
                out["tag_logits"] = torch.log(probs.clamp(1e-6, 1 - 1e-6)
                                              / (1 - probs).clamp_min(1e-6))
            if "genre_logits" in flat:
                probs = torch.softmax(flat["genre_logits"].float(), dim=-1)
                probs = probs.view(b, c, -1).mean(dim=1)
                out["genre_probs"] = probs
                out["genre_logits"] = torch.log(probs.clamp_min(1e-9))
            return out

        out = self._heads(self._encode(x))
        if "tag_logits" in out:
            out["tag_probs"] = torch.sigmoid(out["tag_logits"].float())
        if "genre_logits" in out:
            out["genre_probs"] = torch.softmax(out["genre_logits"].float(), dim=-1)
        return out

    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        decay = [p for _, p in self.named_parameters()
                 if p.requires_grad and p.ndim > 1]
        no_decay = [p for _, p in self.named_parameters()
                    if p.requires_grad and p.ndim <= 1]
        groups = [
            {"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "cnn_decay"},
            {"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "cnn_no_decay"},
        ]
        return [g for g in groups if g["params"]]

    def capacity_report(self) -> dict:
        return {
            "model": "MelCNN (short-chunk)",
            "channels": list(self.channels),
            "n_blocks": len(self.channels),
            "n_mels": self.n_mels,
            "trainable_params": count_parameters(self),
            "total_params": count_parameters(self, trainable_only=False),
        }
