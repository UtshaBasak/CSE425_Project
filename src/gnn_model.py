"""Graph encoders over segment graphs (Task 2), plus the hetero variant.

Depth is deliberately small. These graphs have 4-32 nodes and a diameter of
maybe 4-6 hops, so at three or four rounds of message passing every node's
receptive field is the whole graph and the node states converge to the same
vector -- classic oversmoothing, and it shows up as a *drop* in macro-F1 that
looks like underfitting. Two layers is the default; residual connections and a
final ``mean||max`` readout preserve what individual segments contribute.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .utils import get_logger

LOGGER = get_logger("gbmc.gnn")

__all__ = ["GNNEncoder", "GNNClassifier", "HeteroGNNEncoder", "READOUTS", "CONVS"]

CONVS = ("sage", "gatv2")
READOUTS = ("mean", "max", "mean_max", "attention")
MAX_SANE_LAYERS = 4


class AttentionReadout(nn.Module):
    """Gated attention pooling: a learned score per node, softmaxed per graph."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h, batch):
        logits = self.score(h)                                   # [N, 1]
        # softmax within each graph: subtract the per-graph max for stability
        max_per_graph = global_max_pool(logits, batch)            # [B, 1]
        exp = torch.exp(logits - max_per_graph[batch])
        denom = global_add_pool(exp, batch)[batch] + 1e-8
        weights = exp / denom
        pooled = global_add_pool(h * weights, batch)
        return pooled, weights.squeeze(-1)


class GNNEncoder(nn.Module):
    """Segment-graph encoder returning a graph vector and (optionally) node states.

    ``forward`` returns ``(g, node_h)``; ``node_h`` is ``None`` unless
    ``return_nodes=True``. When ``conv="gatv2"``, the per-edge attention weights
    from the last layer are stashed on ``self.last_attention`` as
    ``(edge_index, alpha)`` so :mod:`src.attention_viz` can render them without
    a second forward pass.
    """

    def __init__(self, in_dim: int = 96, hidden_dim: int = 256, num_layers: int = 2,
                 conv: str = "sage", dropout: float = 0.3, readout: str = "mean_max",
                 residual: bool = True, heads: int = 4, edge_dim: int = 2):
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}, got {conv!r}")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        num_layers = int(num_layers)
        if num_layers > MAX_SANE_LAYERS:
            LOGGER.warning(
                "num_layers=%d on 4-32 node graphs will oversmooth; %d or fewer "
                "is the supported range.", num_layers, MAX_SANE_LAYERS,
            )

        self.conv_name = conv
        self.readout_name = readout
        self.residual = bool(residual)
        self.dropout = float(dropout)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = num_layers
        self.last_attention = None

        self.input_proj = nn.Linear(int(in_dim), int(hidden_dim))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "sage":
                layer = SAGEConv(hidden_dim, hidden_dim)
            else:
                # concat=False keeps the width at hidden_dim so residuals add
                # cleanly; edge_dim lets [is_temporal, cosine] steer attention
                layer = GATv2Conv(hidden_dim, hidden_dim, heads=int(heads),
                                  concat=False, dropout=self.dropout, edge_dim=edge_dim)
            self.convs.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.attn_pool = AttentionReadout(hidden_dim) if readout == "attention" else None
        self.out_dim = hidden_dim * 2 if readout == "mean_max" else hidden_dim

    def forward(self, data, return_nodes: bool = False):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.input_proj(x)
        self.last_attention = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = h
            if self.conv_name == "gatv2":
                h, attn = conv(h, edge_index, edge_attr=edge_attr,
                               return_attention_weights=True)
                self.last_attention = attn
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h = h + residual

        node_weights = None
        if self.readout_name == "mean":
            g = global_mean_pool(h, batch)
        elif self.readout_name == "max":
            g = global_max_pool(h, batch)
        elif self.readout_name == "mean_max":
            g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        else:
            g, node_weights = self.attn_pool(h, batch)
        self.last_node_weights = node_weights
        return (g, h) if return_nodes else (g, None)


class GNNClassifier(nn.Module):
    """Task 2: multi-label tags (and optionally genre) from audio graphs alone.

    Audio-only node features by design -- Task 2's whole point is to show what
    structure buys *before* any text is fused in, so a text leak here would make
    Task 3's ablation uninterpretable.
    """

    def __init__(self, n_tags: int, in_dim: int = 96, hidden_dim: int = 256,
                 num_layers: int = 2, conv: str = "sage", dropout: float = 0.3,
                 readout: str = "mean_max", residual: bool = True, heads: int = 4,
                 n_genres: int = 0, encoder: GNNEncoder | None = None):
        super().__init__()
        self.encoder = encoder or GNNEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, conv=conv,
            dropout=dropout, readout=readout, residual=residual, heads=heads,
        )
        d = self.encoder.out_dim
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, int(n_tags)),
        )
        self.genre_head = nn.Linear(d, int(n_genres)) if n_genres else None
        self.n_tags = int(n_tags)

    def forward(self, data, return_nodes: bool = False):
        g, node_h = self.encoder(data, return_nodes=return_nodes)
        out = {"tag_logits": self.tag_head(self.dropout(g)), "z": g, "node_h": node_h}
        if self.genre_head is not None:
            out["genre_logits"] = self.genre_head(g)
        return out

    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        """No pretrained weights here, so everything gets ``lr_head``."""
        decay = [p for n, p in self.named_parameters()
                 if p.requires_grad and p.ndim > 1 and "LayerNorm" not in n]
        no_decay = [p for n, p in self.named_parameters()
                    if p.requires_grad and (p.ndim <= 1 or "LayerNorm" in n)]
        groups = [
            {"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "gnn_decay"},
            {"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "gnn_no_decay"},
        ]
        return [g for g in groups if g["params"]]


class HeteroGNNEncoder(nn.Module):
    """Bonus: segment + chord heterogeneous encoder over the five edge types.

    Chords enter as a learned embedding of the vocabulary index concatenated
    with their aggregated chroma, so a rare chord symbol still gets a usable
    vector instead of a near-empty one-hot.
    """

    EDGE_TYPES = [
        ("segment", "next", "segment"),
        ("segment", "similar", "segment"),
        ("segment", "contains", "chord"),
        ("chord", "in", "segment"),
        ("chord", "transitions", "chord"),
    ]

    def __init__(self, seg_in_dim: int = 96, chord_in_dim: int = 13,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3,
                 chord_vocab_size: int = 25, chord_embed_dim: int = 32,
                 readout: str = "mean_max"):
        super().__init__()
        from torch_geometric.nn import HeteroConv

        self.chord_embedding = nn.Embedding(int(chord_vocab_size), int(chord_embed_dim))
        self.seg_proj = nn.Linear(int(seg_in_dim), int(hidden_dim))
        self.chord_proj = nn.Linear(int(chord_in_dim) + int(chord_embed_dim), int(hidden_dim))
        self.dropout = float(dropout)
        self.readout_name = readout

        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(
                HeteroConv(
                    {et: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for et in self.EDGE_TYPES},
                    aggr="sum",
                )
            )
        self.norm_segment = nn.LayerNorm(hidden_dim)
        self.norm_chord = nn.LayerNorm(hidden_dim)
        self.out_dim = hidden_dim * 2 if readout == "mean_max" else hidden_dim

    def forward(self, data, return_nodes: bool = False):
        chord_x = data["chord"].x
        chord_ids = getattr(data["chord"], "chord_ids", None)
        if chord_ids is None:
            chord_ids = torch.zeros(chord_x.size(0), dtype=torch.long, device=chord_x.device)
        chord_in = torch.cat([chord_x, self.chord_embedding(chord_ids)], dim=-1)

        h = {
            "segment": self.seg_proj(data["segment"].x),
            "chord": self.chord_proj(chord_in),
        }
        edge_index_dict = {
            et: data[et].edge_index for et in self.EDGE_TYPES if et in data.edge_types
        }
        for layer in self.layers:
            updated = layer(h, edge_index_dict)
            h = {
                key: F.dropout(F.gelu(updated.get(key, value)), p=self.dropout,
                               training=self.training) + value
                for key, value in h.items()
            }
        h["segment"] = self.norm_segment(h["segment"])
        h["chord"] = self.norm_chord(h["chord"])

        seg_batch = getattr(data["segment"], "batch", None)
        if seg_batch is None:
            seg_batch = torch.zeros(h["segment"].size(0), dtype=torch.long,
                                    device=h["segment"].device)
        if self.readout_name == "mean_max":
            g = torch.cat([global_mean_pool(h["segment"], seg_batch),
                           global_max_pool(h["segment"], seg_batch)], dim=-1)
        elif self.readout_name == "max":
            g = global_max_pool(h["segment"], seg_batch)
        else:
            g = global_mean_pool(h["segment"], seg_batch)
        return (g, h) if return_nodes else (g, None)
