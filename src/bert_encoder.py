"""BERT text encoder, the Task 1 tag classifier, and the B3 baseline.

Freezing strategy matters more than it looks on a 4 GB card. Full fine-tuning of
``bert-base`` at batch 8 with a GNN alongside does not fit; and even when it
fits, a 2e-5 BERT and a 1e-3 head trained from step 0 lets the randomly
initialised head emit huge gradients into a pretrained encoder and wash out its
features. So: freeze for ``freeze_epochs``, then unfreeze the top N layers,
which are the semantically specific ones anyway.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .utils import get_logger

LOGGER = get_logger("gbmc.bert")

__all__ = [
    "BertTextEncoder",
    "BertTagClassifier",
    "load_tokenizer",
    "FREEZE_MODES",
]

FREEZE_MODES = ("frozen_probe", "top_n", "full_ft")


def load_tokenizer(model_name: str):
    """Tokenizer for ``model_name``, falling back to a tiny offline vocabulary."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - offline machines
        LOGGER.warning("could not load tokenizer %s (%s); using bert-base-uncased",
                       model_name, exc)
        return AutoTokenizer.from_pretrained("bert-base-uncased")


ATTENTION_NOTE = (
    "transformers defaults to the SDPA attention kernel, which is faster but "
    "returns no attention weights. Anything that needs to *see* the attention "
    "must load the backbone with attn_implementation='eager'."
)


def _encoder_layers(model) -> list[nn.Module]:
    """The transformer blocks, for BERT (``encoder.layer``) or DistilBERT
    (``transformer.layer``)."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return list(model.transformer.layer)
    for attr in ("layers", "layer"):
        block = getattr(model, attr, None)
        if isinstance(block, (nn.ModuleList, list)):
            return list(block)
    return []


class BertTextEncoder(nn.Module):
    """Wraps a HuggingFace encoder and exposes CLS + token states.

    ``forward`` returns ``(cls, tokens)`` where ``tokens`` is ``None`` unless
    ``return_tokens=True`` -- cross-attention fusion needs the full token
    sequence, but Task 1 does not, and materialising ``[B, 128, 768]`` when it
    is not needed is a measurable slice of a 4 GB budget.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 freeze_mode: str = "full_ft", unfreeze_top_n: int = 4,
                 output_attentions: bool = False,
                 gradient_checkpointing: bool = False):
        super().__init__()
        if freeze_mode not in FREEZE_MODES:
            raise ValueError(f"freeze_mode must be one of {FREEZE_MODES}, got {freeze_mode!r}")
        self.model_name = model_name
        self.unfreeze_top_n = int(unfreeze_top_n)
        self.output_attentions = bool(output_attentions)
        self.bert = self._load_backbone(model_name, self.output_attentions)
        self.hidden_size = int(self.bert.config.hidden_size)
        self.freeze_mode = freeze_mode
        self.set_freeze_mode(freeze_mode)
        self.gradient_checkpointing = False
        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # -- construction ----------------------------------------------------- #
    @staticmethod
    def _load_backbone(model_name: str, output_attentions: bool = False):
        from transformers import AutoConfig, AutoModel

        # See ATTENTION_NOTE: SDPA silently returns None for `attentions`, which
        # would leave attention_viz with nothing to plot and no error to explain it.
        kwargs = {"attn_implementation": "eager"} if output_attentions else {}
        try:
            return AutoModel.from_pretrained(model_name, **kwargs)
        except Exception as exc:  # pragma: no cover - offline machines
            LOGGER.warning(
                "could not download %s (%s); falling back to a small randomly "
                "initialised encoder. Fine for smoke tests, NOT for reported results.",
                model_name, exc,
            )
            config = AutoConfig.for_model(
                "distilbert", vocab_size=30522, dim=128, hidden_dim=512,
                n_layers=2, n_heads=4, max_position_embeddings=512,
            )
            return AutoModel.from_config(config, **kwargs)

    # -- freezing --------------------------------------------------------- #
    def set_freeze_mode(self, mode: str) -> None:
        """Apply one of ``frozen_probe`` / ``top_n`` / ``full_ft``."""
        if mode not in FREEZE_MODES:
            raise ValueError(f"unknown freeze_mode {mode!r}")
        self.freeze_mode = mode
        layers = _encoder_layers(self.bert)

        if mode == "full_ft":
            for param in self.bert.parameters():
                param.requires_grad = True
        elif mode == "frozen_probe":
            for param in self.bert.parameters():
                param.requires_grad = False
        else:  # top_n
            for param in self.bert.parameters():
                param.requires_grad = False
            for layer in layers[-max(self.unfreeze_top_n, 0):] if self.unfreeze_top_n else []:
                for param in layer.parameters():
                    param.requires_grad = True
        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        LOGGER.info("BERT freeze_mode=%s -> %d trainable encoder params (of %d layers)",
                    mode, trainable, len(layers))

    def enable_gradient_checkpointing(self, enable: bool = True) -> bool:
        """Trade ~30% speed for a large activation-memory cut.

        Measured on the GTX 1650 (state/vram_report.md): a **no-op below batch
        32**, because the peak there is AdamW optimiser state rather than
        activations, and a 45% cut at batch 64 (3805 -> 2116 MB). Only worth
        enabling once activations actually dominate.

        Note that HuggingFace silently skips checkpointing when the module is in
        eval mode, so this must be paired with ``model.train()`` -- forgetting
        that produces peaks identical to having it off, with no error.
        """
        if not hasattr(self.bert, "gradient_checkpointing_enable"):
            LOGGER.warning("%s does not support gradient checkpointing", self.model_name)
            return False
        if enable:
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.bert, "config"):
                self.bert.config.use_cache = False
        else:
            self.bert.gradient_checkpointing_disable()
        self.gradient_checkpointing = bool(enable)
        LOGGER.info("gradient checkpointing %s for %s",
                    "enabled" if enable else "disabled", self.model_name)
        return True

    def unfreeze_top_layers(self, n: int | None = None) -> None:
        """Convenience for the epoch-``freeze_epochs`` transition in train.py."""
        if n is not None:
            self.unfreeze_top_n = int(n)
        self.set_freeze_mode("top_n")

    # -- forward ---------------------------------------------------------- #
    def forward(self, input_ids, attention_mask, return_tokens: bool = False,
                return_attentions: bool = False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=bool(return_attentions or self.output_attentions),
        )
        hidden = outputs.last_hidden_state              # [B, L, d]
        cls = hidden[:, 0]                              # DistilBERT has no pooler
        tokens = hidden if return_tokens else None
        if return_attentions or self.output_attentions:
            return cls, tokens, getattr(outputs, "attentions", None)
        return cls, tokens

    # -- optimisation ----------------------------------------------------- #
    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        """Two groups: pretrained weights at ``lr_bert``, everything else at ``lr_head``.

        Bias and LayerNorm parameters are excluded from weight decay, which is
        the standard BERT recipe and matters at these small learning rates.
        """
        bert_ids = {id(p) for p in self.bert.parameters()}
        decay, no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            skip_decay = param.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name
            if id(param) in bert_ids:
                (no_decay if skip_decay else decay).append(param)
            else:
                (head_no_decay if skip_decay else head_decay).append(param)
        groups = [
            {"params": decay, "lr": lr_bert, "weight_decay": weight_decay, "name": "bert_decay"},
            {"params": no_decay, "lr": lr_bert, "weight_decay": 0.0, "name": "bert_no_decay"},
            {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_decay"},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_no_decay"},
        ]
        return [g for g in groups if g["params"]]

    # -- Task 4 support --------------------------------------------------- #
    @torch.no_grad()
    def precompute_embeddings(self, texts: Sequence[str], tokenizer=None,
                              batch_size: int = 64, max_length: int = 128,
                              device=None, normalise: bool = False) -> torch.Tensor:
        """Encode a corpus once and return ``[N, d]`` CLS vectors on CPU.

        This is what makes a 512-sample contrastive batch possible on 4 GB: with
        text frozen, the only thing on the GPU per step is the graph encoder and
        an ``[N, d]`` slice of a precomputed matrix.
        """
        tokenizer = tokenizer or load_tokenizer(self.model_name)
        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()

        out = []
        for start in range(0, len(texts), int(batch_size)):
            chunk = [t if isinstance(t, str) and t else "music"
                     for t in texts[start: start + int(batch_size)]]
            encoded = tokenizer(chunk, padding=True, truncation=True,
                                max_length=int(max_length), return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            cls, _ = self.forward(encoded["input_ids"], encoded["attention_mask"])
            if normalise:
                cls = torch.nn.functional.normalize(cls, dim=-1)
            out.append(cls.float().cpu())
        if was_training:
            self.train()
        return torch.cat(out, dim=0) if out else torch.zeros((0, self.hidden_size))


class BertTagClassifier(nn.Module):
    """Task 1 / baseline B3: multi-label tag classification from text alone.

    Emits raw logits; the loss is ``BCEWithLogits`` (masked), and thresholds are
    tuned on validation afterwards. Deliberately *not* softmax -- tags are not
    mutually exclusive, and every metric downstream assumes independent sigmoids.
    """

    def __init__(self, n_tags: int, model_name: str = "distilbert-base-uncased",
                 freeze_mode: str = "full_ft", unfreeze_top_n: int = 4,
                 dropout: float = 0.1, hidden_dim: int | None = None,
                 encoder: BertTextEncoder | None = None,
                 output_attentions: bool = False,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder = encoder or BertTextEncoder(
            model_name, freeze_mode=freeze_mode, unfreeze_top_n=unfreeze_top_n,
            output_attentions=output_attentions,
            gradient_checkpointing=gradient_checkpointing,
        )
        d = self.encoder.hidden_size
        self.dropout = nn.Dropout(dropout)
        if hidden_dim:
            self.head = nn.Sequential(
                nn.Linear(d, int(hidden_dim)), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(int(hidden_dim), int(n_tags)),
            )
        else:
            self.head = nn.Linear(d, int(n_tags))
        self.n_tags = int(n_tags)

    def forward(self, input_ids, attention_mask, return_attn: bool = False):
        if return_attn:
            cls, _, attentions = self.encoder(
                input_ids, attention_mask, return_tokens=False, return_attentions=True
            )
            return self.head(self.dropout(cls)), attentions
        cls, _ = self.encoder(input_ids, attention_mask)
        return self.head(self.dropout(cls))

    def param_groups(self, lr_bert: float, lr_head: float,
                     weight_decay: float = 0.01) -> list[dict]:
        bert_ids = {id(p) for p in self.encoder.bert.parameters()}
        decay, no_decay, head_decay, head_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            skip_decay = param.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name
            if id(param) in bert_ids:
                (no_decay if skip_decay else decay).append(param)
            else:
                (head_no_decay if skip_decay else head_decay).append(param)
        groups = [
            {"params": decay, "lr": lr_bert, "weight_decay": weight_decay, "name": "bert_decay"},
            {"params": no_decay, "lr": lr_bert, "weight_decay": 0.0, "name": "bert_no_decay"},
            {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_decay"},
            {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_no_decay"},
        ]
        return [g for g in groups if g["params"]]
