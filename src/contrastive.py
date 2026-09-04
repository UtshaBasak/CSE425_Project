"""Task 4: contrastive dual encoder for audio-graph <-> caption retrieval.

Contrastive learning wants large batches -- the in-batch negatives *are* the
training signal, and R@10 improves markedly from 64 to 512. A 4 GB card cannot
hold 512 BERT forward passes. The way out is ``contrastive.precompute_text``:
freeze the text encoder, encode the caption corpus once into an ``[N, d]``
matrix on CPU, and let each step slice rows out of it. Only the graph encoder
then needs activations, and 512 fits comfortably.

The temperature is stored as a *log* scale and clamped, following CLIP: an
unclamped learnable temperature drifts toward zero, the logits blow up, and fp16
training produces ``inf`` within a few hundred steps.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger

LOGGER = get_logger("gbmc.contrastive")

__all__ = [
    "DualEncoder",
    "symmetric_info_nce",
    "build_similarity_matrix",
    "zero_shot_tag",
    "DEFAULT_TEMPLATES",
]

DEFAULT_TEMPLATES: list[str] = [
    "{tag}",
    "a recording of {tag} music",
    "this music sounds {tag}",
    "a track featuring {tag}",
]


class DualEncoder(nn.Module):
    """Graph encoder + text encoder projected into one L2-normalised space."""

    def __init__(self, gnn, bert, shared_dim: int = 256, temperature_init: float = 0.07,
                 learnable: bool = True, clamp_logit_scale: float = 100.0,
                 dropout: float = 0.1):
        super().__init__()
        self.gnn = gnn
        self.bert = bert
        self.shared_dim = int(shared_dim)
        self.clamp_logit_scale = float(clamp_logit_scale)

        gnn_dim = getattr(gnn, "out_dim", shared_dim)
        bert_dim = getattr(bert, "hidden_size", shared_dim)
        self.graph_proj = nn.Sequential(
            nn.LayerNorm(gnn_dim), nn.Linear(gnn_dim, self.shared_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(self.shared_dim, self.shared_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(bert_dim), nn.Linear(bert_dim, self.shared_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(self.shared_dim, self.shared_dim),
        )

        init = float(np.log(1.0 / max(float(temperature_init), 1e-4)))
        scale = torch.tensor(init, dtype=torch.float32)
        self.logit_scale = nn.Parameter(scale) if learnable else nn.Parameter(
            scale, requires_grad=False
        )
        self._precomputed_text: torch.Tensor | None = None

    # -- temperature ------------------------------------------------------ #
    def get_logit_scale(self) -> torch.Tensor:
        """``exp(log_scale)``, clamped so fp16 logits stay finite."""
        return self.logit_scale.clamp(max=float(np.log(self.clamp_logit_scale))).exp()

    @property
    def temperature(self) -> float:
        return float(1.0 / self.get_logit_scale().detach().cpu())

    # -- encoders --------------------------------------------------------- #
    def encode_graph(self, data) -> torch.Tensor:
        g, _ = self.gnn(data, return_nodes=False)
        return F.normalize(self.graph_proj(g), dim=-1)

    def encode_text(self, ids, mask) -> torch.Tensor:
        cls, _ = self.bert(ids, mask)
        return F.normalize(self.text_proj(cls), dim=-1)

    # -- the 4 GB path ---------------------------------------------------- #
    def set_precomputed_text(self, embeddings: torch.Tensor) -> None:
        """Install an ``[N, bert_dim]`` matrix of frozen CLS vectors.

        Kept on CPU; rows are moved to the GPU per step. Storing the whole
        matrix on a 4 GB card would defeat the purpose.
        """
        self._precomputed_text = embeddings.detach().float().cpu()
        for param in self.bert.parameters():
            param.requires_grad = False
        LOGGER.info("precomputed %d text embeddings; BERT frozen for Task 4",
                    embeddings.shape[0])

    def encode_text_precomputed(self, indices) -> torch.Tensor:
        """Project frozen CLS rows selected by integer index."""
        if self._precomputed_text is None:
            raise RuntimeError("call set_precomputed_text() first")
        device = next(self.text_proj.parameters()).device
        idx = torch.as_tensor(indices, dtype=torch.long)
        rows = self._precomputed_text.index_select(0, idx.cpu()).to(device)
        return F.normalize(self.text_proj(rows), dim=-1)

    def forward(self, data, ids=None, mask=None, text_indices=None):
        g = self.encode_graph(data)
        if text_indices is not None:
            t = self.encode_text_precomputed(text_indices)
        else:
            t = self.encode_text(ids, mask)
        return g, t, self.get_logit_scale()

    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        bert_params = {id(p) for p in self.bert.parameters()}
        buckets = {"bert_decay": [], "bert_no_decay": [], "head_decay": [], "head_no_decay": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            skip_decay = param.ndim <= 1 or name.endswith(".bias") or "LayerNorm" in name
            prefix = "bert" if id(param) in bert_params else "head"
            buckets[f"{prefix}_{'no_decay' if skip_decay else 'decay'}"].append(param)
        lrs = {"bert": lr_bert, "head": lr_head}
        groups = []
        for key, params in buckets.items():
            if not params:
                continue
            groups.append({
                "params": params,
                "lr": lrs[key.split("_")[0]],
                "weight_decay": 0.0 if "no_decay" in key else weight_decay,
                "name": key,
            })
        return groups


def symmetric_info_nce(g: torch.Tensor, t: torch.Tensor, logit_scale) -> torch.Tensor:
    """Averaged graph->text and text->graph InfoNCE over in-batch negatives.

    Symmetric because the two directions are different tasks: "given this audio,
    find the caption" and "given this caption, find the audio". Optimising only
    the first gives a model that is good at exactly one of the two columns of
    the retrieval table, which is the classic dual-encoder failure.
    """
    if g.shape[0] != t.shape[0]:
        raise ValueError(f"batch mismatch: {g.shape[0]} graphs vs {t.shape[0]} texts")
    if g.shape[0] < 2:
        # a single pair has no negatives; the loss is degenerate but finite
        return torch.zeros((), device=g.device, dtype=torch.float32)

    g = F.normalize(g.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    scale = logit_scale if torch.is_tensor(logit_scale) else torch.tensor(
        float(logit_scale), device=g.device
    )
    logits = scale * g @ t.t()
    labels = torch.arange(g.shape[0], device=g.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def build_similarity_matrix(g_emb, t_emb) -> np.ndarray:
    """Cosine similarity ``[n_graphs, n_texts]``; rows query, columns gallery."""
    g = _to_tensor(g_emb).float()
    t = _to_tensor(t_emb).float()
    g = F.normalize(g, dim=-1)
    t = F.normalize(t, dim=-1)
    return (g @ t.t()).detach().cpu().numpy()


@torch.no_grad()
def zero_shot_tag(g_emb, tag_names: Sequence[str], bert, templates: Sequence[str] | None = None,
                  tokenizer=None, dual: "DualEncoder | None" = None, device=None,
                  max_length: int = 128, top_k: int = 5) -> dict:
    """Score graphs against tag names via prompt templates, and report the spread.

    Several templates are used on purpose. Zero-shot scores move a lot with
    phrasing, and a single-template number invites the reader to believe a
    precision the method does not have; ``template_spread`` is the honest
    companion to the headline.
    """
    from .bert_encoder import load_tokenizer

    templates = list(templates) if templates else list(DEFAULT_TEMPLATES)
    if len(templates) < 3:
        raise ValueError("use at least three templates so the spread is meaningful")

    tokenizer = tokenizer or load_tokenizer(getattr(bert, "model_name", "distilbert-base-uncased"))
    device = device or next(bert.parameters()).device
    g = F.normalize(_to_tensor(g_emb).float().to(device), dim=-1)

    per_template = []
    for template in templates:
        prompts = [template.format(tag=tag) for tag in tag_names]
        encoded = tokenizer(prompts, padding=True, truncation=True,
                            max_length=int(max_length), return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if dual is not None:
            t = dual.encode_text(encoded["input_ids"], encoded["attention_mask"])
        else:
            cls, _ = bert(encoded["input_ids"], encoded["attention_mask"])
            t = F.normalize(cls.float(), dim=-1)
        if t.shape[-1] != g.shape[-1]:
            raise ValueError(
                f"tag embeddings are {t.shape[-1]}-dim but graph embeddings are "
                f"{g.shape[-1]}-dim; pass the DualEncoder via `dual=` so both go "
                "through the shared projection"
            )
        per_template.append((g @ t.t()).cpu().numpy())

    stacked = np.stack(per_template, axis=0)              # [T, n_graphs, n_tags]
    mean_scores = stacked.mean(axis=0)
    spread = float(stacked.std(axis=0).mean())
    order = np.argsort(-mean_scores, axis=1)[:, : int(top_k)]

    return {
        "scores": mean_scores,
        "per_template_scores": stacked,
        "templates": templates,
        "tag_names": list(tag_names),
        "template_spread": spread,
        "top_k_indices": order,
        "top_k_tags": [[tag_names[j] for j in row] for row in order],
    }


def _to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(np.asarray(x))
