"""Shared fixtures.

The smoke tests need a transformer that is *small* -- running distilbert through
seven fusion variants on CPU turns a 40-second test suite into a 20-minute one.
Rather than depending on a tiny checkpoint from the Hub (whose availability and
tokenizer format drift between transformers releases), we build one locally: the
real ``distilbert-base-uncased`` tokenizer paired with a 2-layer, 64-dim
randomly initialised body. Weights are meaningless, which is the point -- these
tests check plumbing, not accuracy.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory) -> str:
    """Path to a small locally built encoder, usable as ``bert.model_name``."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    out = tmp_path_factory.mktemp("tiny_bert")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.save_pretrained(out)

    config = AutoConfig.for_model(
        "distilbert",
        vocab_size=tokenizer.vocab_size,
        dim=64,
        hidden_dim=128,
        n_layers=2,
        n_heads=2,
        max_position_embeddings=512,
    )
    AutoModel.from_config(config).save_pretrained(out)
    return str(out).replace("\\", "/")
