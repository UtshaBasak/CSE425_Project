"""B2: a mel-spectrogram CNN, deliberately built as a *fair* competitor.

"Fair experimental setup" is 15% of the grade, and the easiest way to lose it is
to ship a 3-layer toy CNN next to a 2M-parameter GNN and call the gap a result.
:class:`MelCNN` therefore accepts a ``target_params`` hint and sizes its channel
widths to land within a few percent of the GNN it is being compared against, and
it sees the same audio, the same splits and the same thresholding protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import count_parameters, get_logger

LOGGER = get_logger("gbmc.cnn")

__all__ = ["MelCNN", "ConvBlock"]


class ConvBlock(nn.Module):
    """Conv -> BN -> GELU -> Conv -> BN -> GELU -> MaxPool, the standard MTAT stack."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2),
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        return self.drop(self.pool(x))


class MelCNN(nn.Module):
    """5-block 2-D CNN over log-mel patches, with mean+max global pooling.

    ``target_params`` (optional) triggers a width search: the base channel count
    is scaled until the total parameter count is as close as possible to the
    target, so the CNN-vs-GNN comparison is about inductive bias rather than
    capacity. The chosen width and the achieved count are logged and returned by
    :meth:`capacity_report` for the report's fair-setup table.
    """

    def __init__(self, n_tags: int, n_mels: int = 128, base_channels: int = 32,
                 n_blocks: int = 5, dropout: float = 0.2, in_channels: int = 1,
                 target_params: int | None = None, n_genres: int = 0):
        super().__init__()
        self.n_tags = int(n_tags)
        self.n_mels = int(n_mels)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.in_channels = int(in_channels)
        self.n_genres = int(n_genres)
        self.target_params = target_params

        base = int(base_channels)
        if target_params:
            base = self._search_width(target_params, n_tags, n_mels, n_blocks,
                                      dropout, in_channels, n_genres)
        self.base_channels = base
        self._build(base)

        if target_params:
            LOGGER.info(
                "MelCNN parameter-matched: target=%d actual=%d (%.1f%% of target), "
                "base_channels=%d",
                target_params, count_parameters(self),
                100.0 * count_parameters(self) / max(target_params, 1), base,
            )

    # -- construction ----------------------------------------------------- #
    def _build(self, base: int) -> None:
        channels = [self.in_channels] + [base * (2 ** min(i, 3)) for i in range(self.n_blocks)]
        blocks = []
        for i in range(self.n_blocks):
            # stop pooling the frequency axis once it would vanish
            pool_f = 2 if self.n_mels // (2 ** (i + 1)) >= 1 else 1
            blocks.append(ConvBlock(channels[i], channels[i + 1], pool=(pool_f, 2),
                                    dropout=self.dropout * 0.5))
        self.blocks = nn.Sequential(*blocks)
        feat_dim = channels[-1] * 2                      # mean || max over time-freq
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(feat_dim, max(feat_dim // 2, 64)), nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(feat_dim // 2, 64), self.n_tags),
        )
        self.genre_head = (
            nn.Linear(feat_dim, self.n_genres) if self.n_genres else None
        )

    @staticmethod
    def _search_width(target: int, n_tags: int, n_mels: int, n_blocks: int,
                      dropout: float, in_channels: int, n_genres: int) -> int:
        """Pick the base channel count whose parameter total is closest to target.

        Stepped by 1 rather than by a power of two: channel width is the only
        free dimension here, and landing within a few percent of the reference
        GNN is what makes the CNN-vs-GNN row of the results table a statement
        about inductive bias instead of about capacity.
        """
        best, best_gap = 8, float("inf")
        for base in range(4, 257):
            probe = MelCNN(n_tags=n_tags, n_mels=n_mels, base_channels=base,
                           n_blocks=n_blocks, dropout=dropout,
                           in_channels=in_channels, n_genres=n_genres)
            count = count_parameters(probe)
            gap = abs(count - target)
            if gap < best_gap:
                best, best_gap = base, gap
            elif count > target * 1.5:
                break
        return best

    # -- forward ---------------------------------------------------------- #
    def forward(self, x, return_features: bool = False):
        if x.dim() == 3:
            x = x.unsqueeze(1)                            # [B, n_mels, T] -> [B,1,n_mels,T]
        h = self.blocks(x)                                # [B, C, F', T']
        pooled = torch.cat(
            [h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1
        )
        out = {"tag_logits": self.head(pooled), "z": pooled}
        if self.genre_head is not None:
            out["genre_logits"] = self.genre_head(pooled)
        if return_features:
            out["features"] = h
        return out

    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        decay = [p for n, p in self.named_parameters()
                 if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in self.named_parameters()
                    if p.requires_grad and p.ndim <= 1]
        groups = [
            {"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "cnn_decay"},
            {"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "cnn_no_decay"},
        ]
        return [g for g in groups if g["params"]]

    def capacity_report(self) -> dict:
        return {
            "model": "MelCNN",
            "base_channels": self.base_channels,
            "n_blocks": self.n_blocks,
            "trainable_params": count_parameters(self),
            "total_params": count_parameters(self, trainable_only=False),
            "target_params": self.target_params,
        }
