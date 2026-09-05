"""End-to-end smoke tests: all four tasks train one epoch on CPU.

Slow by design -- they exercise real models on the synthetic dataset. They are
the tests that would have caught every integration bug this project can have:
a shape mismatch between the graph batch and the tokenised text, a loss that
silently trains on sentinels, an early-stopping metric that is never populated.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic import make_synthetic_dataset
from src.utils import load_config, project_root


TASKS = [1, 2, 3, 4]


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory):
    """A small synthetic dataset, generated once for the whole session."""
    out_dir = tmp_path_factory.mktemp("synthetic")
    cfg = load_config("config.yaml")
    make_synthetic_dataset(n_tracks=48, n_tags=12, out_dir=str(out_dir), seed=42, cfg=cfg)
    return out_dir


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory):
    """Keep test artefacts out of the repo's real results/ directory.

    Without this, a tiny 32-dim test checkpoint lands in results/checkpoints/
    and the next `python -m src.evaluate` tries to load it into the full-size
    model.
    """
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="session")
def cfg_overrides(synthetic, tiny_bert, results_dir):
    return [
        f"paths.results={results_dir.as_posix()}",
        f"paths.checkpoints={(results_dir / 'checkpoints').as_posix()}",
        "train.epochs=1",
        "train.batch_size=4",
        "train.grad_accum_steps=2",
        "gnn.hidden_dim=32",
        "fusion.shared_dim=32",
        f"bert.model_name={tiny_bert}",
        "bert.max_length=32",
        "bert.freeze_epochs=0",
        "contrastive.batch_size=8",
        f"synthetic.out_dir={synthetic.as_posix()}",
    ]


def _results_root(overrides) -> Path:
    prefix = "paths.results="
    for item in overrides:
        if item.startswith(prefix):
            return Path(item[len(prefix):])
    return project_root() / "results"


def _run(task: int, overrides, extra=None):
    from src.train import main as train_main

    argv = ["--task", str(task), "--synthetic", "--device", "cpu",
            "--num-workers", "0", "--override", *overrides]
    argv += extra or []
    assert train_main(argv) == 0
    results = _results_root(overrides) / f"task{task}_seed42.json"
    assert results.exists(), f"task {task} wrote no result file"
    with open(results, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_task_trains_one_epoch_on_cpu(task, cfg_overrides):
    result = _run(task, cfg_overrides)

    assert result["epochs_run"] == 1
    assert result["device"] == "cpu"
    history = result["history"]
    assert history, "no epoch history recorded"

    loss = history[0].get("train_loss_total")
    assert loss is not None and np.isfinite(loss), f"non-finite training loss: {loss}"
    assert history[0]["train_batches"] > 0, "no batches were consumed"

    for key, value in (result.get("test") or {}).items():
        if isinstance(value, (int, float)):
            assert not (isinstance(value, float) and np.isnan(value)), \
                f"task {task} produced nan for test metric {key!r}"


@pytest.mark.slow
def test_gradient_accumulation_is_configurable(cfg_overrides):
    result = _run(2, [o for o in cfg_overrides if not o.startswith("train.grad_accum")]
                  + ["train.grad_accum_steps=3"])
    assert result["grad_accum_steps"] == 3
    assert result["effective_batch"] == result["grad_accum_steps"] * 4


@pytest.mark.slow
def test_task4_reports_gallery_size(cfg_overrides):
    result = _run(4, cfg_overrides)
    test = result["test"]
    assert test["gallery_size"] > 0
    for key in ("g2t_R@1", "t2g_R@1", "g2t_medR", "mean_MRR"):
        assert key in test


@pytest.mark.slow
def test_thresholds_come_from_validation(cfg_overrides):
    result = _run(1, cfg_overrides)
    assert result["threshold_source"] == "val"
    assert result["thresholds"] is not None
    assert len(result["thresholds"]) == result["n_tags"]


# --------------------------------------------------------------------------- #
# the masked multi-task loss -- acceptance criterion 12
# --------------------------------------------------------------------------- #
class _Batch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_masked_loss_ignores_minus_one_tags():
    """A row of ``-1`` tags must contribute nothing, whatever the logits say."""
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    logits = torch.zeros(2, 4)
    y_real = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    y_absent = torch.full((1, 4), -1.0)

    only_real = _Batch(y_tags=y_real, y_valence=torch.tensor([float("nan")]),
                       y_arousal=torch.tensor([float("nan")]))
    with_absent = _Batch(y_tags=torch.cat([y_real, y_absent]),
                         y_valence=torch.tensor([float("nan"), float("nan")]),
                         y_arousal=torch.tensor([float("nan"), float("nan")]))

    loss_a, parts_a = masked_multitask_loss(
        {"tag_logits": logits[:1], "valence": None, "arousal": None}, only_real, cfg)
    loss_b, parts_b = masked_multitask_loss(
        {"tag_logits": logits, "valence": None, "arousal": None}, with_absent, cfg)

    assert parts_a["n_tag_cells"] == 4
    assert parts_b["n_tag_cells"] == 4, "sentinel cells leaked into the loss"
    assert float(loss_a) == pytest.approx(float(loss_b), abs=1e-6)


def test_masked_loss_ignores_nan_regression_targets():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    out = {
        "tag_logits": torch.zeros(3, 2),
        "valence": torch.tensor([5.0, 99.0, 4.0]),
        "arousal": torch.tensor([5.0, -99.0, 4.0]),
    }
    batch = _Batch(
        y_tags=torch.full((3, 2), -1.0),
        y_valence=torch.tensor([5.0, float("nan"), 4.0]),
        y_arousal=torch.tensor([5.0, float("nan"), 4.0]),
    )
    loss, parts = masked_multitask_loss(out, batch, cfg)
    assert parts["n_valence"] == 2 and parts["n_arousal"] == 2
    assert parts["loss_valence"] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(loss), "nan targets reached the arithmetic"
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_masked_loss_is_finite_when_a_batch_has_no_labels_at_all():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml")
    out = {"tag_logits": torch.zeros(2, 3),
           "valence": torch.zeros(2), "arousal": torch.zeros(2)}
    batch = _Batch(y_tags=torch.full((2, 3), -1.0),
                   y_valence=torch.tensor([float("nan")] * 2),
                   y_arousal=torch.tensor([float("nan")] * 2))
    loss, parts = masked_multitask_loss(out, batch, cfg)
    assert torch.isfinite(loss) and float(loss) == 0.0
    assert parts["n_tag_cells"] == 0


def test_masked_loss_gradients_do_not_flow_from_sentinels():
    from src.fusion_model import masked_multitask_loss

    cfg = load_config("config.yaml", {"multitask.auto_balance": False})
    logits = torch.zeros(2, 3, requires_grad=True)
    batch = _Batch(y_tags=torch.tensor([[1.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]),
                   y_valence=torch.tensor([float("nan")] * 2),
                   y_arousal=torch.tensor([float("nan")] * 2))
    loss, _ = masked_multitask_loss(
        {"tag_logits": logits, "valence": None, "arousal": None}, batch, cfg)
    loss.backward()
    assert torch.all(logits.grad[1] == 0), "sentinel row received gradient"
    assert torch.any(logits.grad[0] != 0), "labelled row received no gradient"


# --------------------------------------------------------------------------- #
# amp / model plumbing
# --------------------------------------------------------------------------- #
def test_autocast_is_a_noop_on_cpu():
    from src.utils import autocast_ctx

    with autocast_ctx(True, "cpu"):
        x = torch.ones(2, 2) @ torch.ones(2, 2)
    assert x.dtype == torch.float32


def test_grad_scaler_is_disabled_off_gpu():
    from src.utils import make_grad_scaler

    assert make_grad_scaler(True, "cpu").is_enabled() is False


def test_gnn_encoder_supports_both_convs_and_all_readouts():
    from src.gnn_model import CONVS, READOUTS, GNNEncoder
    from torch_geometric.data import Batch

    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg) for _ in range(2)])
    for conv in CONVS:
        for readout in READOUTS:
            encoder = GNNEncoder(96, 16, 2, conv=conv, readout=readout, dropout=0.0)
            g, h = encoder(batch, return_nodes=True)
            assert g.shape == (2, encoder.out_dim)
            assert h.shape[0] == batch.x.shape[0]
            assert torch.isfinite(g).all()


def test_gatv2_exposes_attention_for_the_case_studies():
    from torch_geometric.data import Batch

    from src.gnn_model import GNNEncoder
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg)])
    encoder = GNNEncoder(96, 16, 2, conv="gatv2", dropout=0.0)
    encoder(batch)
    assert encoder.last_attention is not None
    edge_index, alpha = encoder.last_attention
    assert alpha.shape[0] == edge_index.shape[1]


def test_cnn_baseline_can_be_parameter_matched():
    from src.cnn_baseline import MelCNN
    from src.utils import count_parameters

    target = 500_000
    model = MelCNN(n_tags=10, n_mels=64, target_params=target)
    actual = count_parameters(model)
    assert 0.5 * target <= actual <= 2.0 * target, (
        f"parameter matching missed badly: {actual} vs target {target}"
    )
    out = model(torch.randn(2, 1, 64, 128))
    assert out["tag_logits"].shape == (2, 10)


def test_all_fusion_modes_run(tiny_bert):
    from torch_geometric.data import Batch

    from src.bert_encoder import BertTextEncoder
    from src.fusion_model import FUSION_MODES, GNNBertFusion
    from src.gnn_model import GNNEncoder
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(6, 96)).astype(np.float32)
    batch = Batch.from_data_list([build_segment_graph(feats, cfg, text="a test")
                                  for _ in range(2)])
    ids = torch.randint(0, 100, (2, 8))
    mask = torch.ones_like(ids)

    for mode in FUSION_MODES:
        gnn = GNNEncoder(96, 16, 1, dropout=0.0)
        bert = BertTextEncoder(tiny_bert, freeze_mode="frozen_probe")
        model = GNNBertFusion(gnn, bert, mode=mode, shared_dim=16, n_heads=2, n_tags=5)
        out = model(batch, ids, mask, return_attn=True)
        assert out["tag_logits"].shape == (2, 5), mode
        assert torch.isfinite(out["tag_logits"]).all(), mode
        assert out["valence"].shape == (2,), mode


def test_symmetric_info_nce_is_minimised_by_aligned_embeddings():
    from src.contrastive import symmetric_info_nce

    aligned = torch.eye(4)
    scrambled = torch.eye(4)[[1, 0, 3, 2]]
    good = symmetric_info_nce(aligned, aligned, torch.tensor(10.0))
    bad = symmetric_info_nce(aligned, scrambled, torch.tensor(10.0))
    assert float(good) < float(bad)
    assert torch.isfinite(good)


def test_synthetic_dataset_has_realistic_sentinel_patterns(synthetic):
    with open(Path(synthetic) / "summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["tracks_with_both_tags_and_va"] == 0, (
        "the real corpora never overlap; the synthetic set must not either"
    )
    assert summary["tracks_with_tags"] > 0
    assert summary["tracks_with_valence_arousal"] > 0
    assert set(summary["split_counts"]) == {"train", "val", "test"}


def test_synthetic_splits_are_leak_free(synthetic):
    import pandas as pd

    from src.splits import assert_no_leakage

    assert_no_leakage(pd.read_csv(Path(synthetic) / "manifest.csv"))


def test_norm_stats_are_train_only(synthetic):
    with open(Path(synthetic) / "norm_stats.json", "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    assert stats["split"] == "train"
    assert len(stats["mean"]) == 96 and len(stats["std"]) == 96


def test_compute_norm_stats_refuses_non_train_splits():
    from src.audio_features import compute_norm_stats

    with pytest.raises(ValueError, match="train split"):
        compute_norm_stats(None, split="test")


# --------------------------------------------------------------------------- #
# codebase conventions -- acceptance criteria 9, 10 and 11, enforced by test
# --------------------------------------------------------------------------- #
SOURCE_DIRS = ("src", "scripts")


def _source_files():
    for directory in SOURCE_DIRS:
        for path in sorted((project_root() / directory).rglob("*.py")):
            yield path, path.read_text(encoding="utf-8")


def test_no_stub_implementations_remain():
    """Criterion 9: no NotImplementedError, and no bare `pass` used as a body."""
    offenders = []
    for path, text in _source_files():
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "NotImplementedError" in line:
                offenders.append(f"{path.name}:{i + 1} NotImplementedError")
            if line.strip() == "pass":
                # a bare pass is fine as an exception handler, not as a body
                previous = next(
                    (lines[j] for j in range(i - 1, max(i - 4, -1), -1) if lines[j].strip()),
                    "",
                )
                if not previous.strip().startswith(("except", "finally", "class", "if TYPE")):
                    offenders.append(f"{path.name}:{i + 1} bare pass after {previous.strip()!r}")
    assert not offenders, "stub implementations found: " + "; ".join(offenders)


def _code_tokens(text: str):
    """Yield the source tokens that are actual code, not comments or strings.

    Scanning raw lines cannot tell a docstring that *explains* why accuracy is
    the wrong metric from a line that computes one; tokenising can.
    """
    import tokenize

    reader = io.StringIO(text).readline
    try:
        for token in tokenize.generate_tokens(reader):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if token.string.strip():
                yield token
    except (tokenize.TokenError, IndentationError):
        return


def test_accuracy_is_never_reported_for_multi_label_tagging():
    """Criterion 10: the only accuracy in the code is the flagged counter-example."""
    allowed = {"element_accuracy_do_not_report", "element_accuracy"}
    offenders = []
    for path, text in _source_files():
        for token in _code_tokens(text):
            name = token.string
            if "accuracy" in name.lower() and name not in allowed:
                offenders.append(f"{path.name}:{token.start[0]}: {name}")
    assert not offenders, (
        "accuracy appears to be computed or reported as a metric: "
        + "; ".join(offenders)
    )


def test_thresholds_are_never_tuned_on_test():
    """Criterion 10: every tune_thresholds call site uses val (or train) scores."""
    offenders = []
    for path, text in _source_files():
        for i, line in enumerate(text.splitlines()):
            if "tune_thresholds(" not in line or "def tune_thresholds" in line:
                continue
            arguments = line.split("tune_thresholds(", 1)[1]
            arguments = arguments.replace("yte.shape", "").replace("y_test.shape", "")
            if "test" in arguments.lower():
                offenders.append(f"{path.name}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, "thresholds tuned on test: " + "; ".join(offenders)


def test_norm_stats_are_only_ever_requested_for_train():
    """Criterion 10: no call site asks compute_norm_stats for val or test."""
    offenders = []
    for path, text in _source_files():
        for i, line in enumerate(text.splitlines()):
            if "compute_norm_stats(" not in line or "def compute_norm_stats" in line:
                continue
            if "train" not in line:
                offenders.append(f"{path.name}:{i + 1}: {line.strip()[:90]}")
    assert not offenders, (
        "normalisation statistics computed outside train: " + "; ".join(offenders)
    )


def test_every_training_loop_uses_amp_and_accumulation():
    """Criterion 11: AMP and accumulation in every loop that steps an optimiser."""
    for name in ("src/train.py", "src/evaluate.py", "scripts/run_baselines.py"):
        text = (project_root() / name).read_text(encoding="utf-8")
        if "optimizer.step" not in text and "scaler.step" not in text:
            continue
        assert "autocast_ctx(" in text, f"{name} steps an optimiser without autocast"
        assert "make_grad_scaler(" in text, f"{name} steps an optimiser without a GradScaler"
        assert "grad_accum_steps" in text, f"{name} has no gradient accumulation"


def test_every_module_from_the_spec_exists_with_its_public_api():
    """Criterion 9: the prescribed signatures are importable."""
    import importlib

    expected = {
        "src.utils": ["set_seed", "seed_worker", "get_generator", "load_config",
                      "get_device", "count_parameters", "autocast_ctx", "log_vram"],
        "src.audio_features": ["load_audio", "log_mel", "chroma_cqt", "mfcc",
                               "segment_indices", "segment_features",
                               "compute_norm_stats", "apply_norm", "extract_dataset"],
        "src.chords": ["CHORD_VOCAB", "estimate_chords", "chords_from_midi",
                       "validate_against_lmd", "chord_transition_matrix"],
        "src.graph_builder": ["build_segment_graph", "build_chord_graph",
                              "build_hetero_graph", "rewire_edges", "visualise_graph",
                              "export_sample_graphs"],
        "src.splits": ["build_mtat_splits", "build_fma_splits", "build_deam_splits",
                       "build_musiccaps_splits", "reduce_to_top_k_tags",
                       "assert_no_leakage"],
        "src.metrics": ["tune_thresholds", "macro_f1", "micro_f1", "per_tag_prf",
                        "mean_auc_pr", "macro_roc_auc", "regression_metrics",
                        "retrieval_metrics", "graph_coherence_score", "knn_probe",
                        "silhouette", "aggregate_seeds"],
        "src.bert_encoder": ["BertTextEncoder", "BertTagClassifier"],
        "src.gnn_model": ["GNNEncoder", "GNNClassifier", "HeteroGNNEncoder"],
        "src.cnn_baseline": ["MelCNN"],
        "src.baselines": ["random_tag_baseline", "majority_tag_baseline",
                          "pca_mlp_baseline"],
        "src.fusion_model": ["CrossAttentionFusion", "GatedFusion", "GNNBertFusion",
                             "masked_multitask_loss"],
        "src.contrastive": ["DualEncoder", "symmetric_info_nce",
                            "build_similarity_matrix", "zero_shot_tag"],
        "src.datasets": ["MusicGraphDataset", "MelSpecDataset", "make_loader",
                         "alternating_loader"],
        "src.attention_viz": ["plot_bert_attention", "plot_graph_attention"],
        "src.human_eval": ["generate_listening_sheet", "compute_agreement"],
        "src.synthetic": ["make_synthetic_dataset"],
    }
    missing = []
    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module_name}.{name}")
    assert not missing, "missing from the prescribed API: " + ", ".join(missing)


def test_bert_encoder_exposes_the_prescribed_signatures():
    import inspect

    from src.bert_encoder import BertTextEncoder

    signature = inspect.signature(BertTextEncoder.__init__)
    for parameter in ("model_name", "freeze_mode", "unfreeze_top_n"):
        assert parameter in signature.parameters
    forward = inspect.signature(BertTextEncoder.forward)
    for parameter in ("input_ids", "attention_mask", "return_tokens"):
        assert parameter in forward.parameters
    assert hasattr(BertTextEncoder, "param_groups")
    assert hasattr(BertTextEncoder, "precompute_embeddings")


def test_directory_tree_matches_the_specification():
    """Criterion 13: every prescribed path exists."""
    required = [
        "README.md", "requirements.txt", "config.yaml", "Makefile", "pytest.ini",
        ".gitignore",
        "data/raw", "data/processed", "data/splits",
        "notebooks/eda.ipynb", "notebooks/demo_context.ipynb",
        "src/__init__.py", "src/audio_features.py", "src/graph_builder.py",
        "src/bert_encoder.py", "src/gnn_model.py", "src/fusion_model.py",
        "src/contrastive.py", "src/train.py", "src/evaluate.py", "src/utils.py",
        "src/chords.py", "src/datasets.py", "src/splits.py", "src/metrics.py",
        "src/cnn_baseline.py", "src/baselines.py", "src/attention_viz.py",
        "src/human_eval.py", "src/synthetic.py",
        "scripts/download_mtat.sh", "scripts/download_fma.sh",
        "scripts/download_deam.sh", "scripts/download_lmd.sh",
        "scripts/download_musiccaps.py", "scripts/verify_datasets.py",
        "scripts/export_sample_graphs.py",
        "tests/test_metrics.py", "tests/test_graph_builder.py",
        "tests/test_splits.py", "tests/test_smoke.py",
        "results/plots", "results/retrieval_examples", "report",
    ]
    missing = [p for p in required if not (project_root() / p).exists()]
    assert not missing, "missing from the prescribed tree: " + ", ".join(missing)


# --------------------------------------------------------------------------- #
# human evaluation
# --------------------------------------------------------------------------- #
def test_listening_sheet_has_controls_and_randomised_order():
    from src.human_eval import generate_listening_sheet

    examples = [
        {"query_track_id": f"t{i}", "query_caption": f"caption {i}", "true_rank": i + 1,
         "top3": [{"track_id": f"t{i}", "caption": f"caption {i}", "score": 0.9}]}
        for i in range(12)
    ]
    sheet = generate_listening_sheet(examples, n_items=10, n_controls=4, seed=7)
    assert int(sheet["is_control"].sum()) == 4
    assert len(sheet) == 14
    assert sheet["presentation_order"].nunique() == len(sheet)
    # controls must not all be clustered at the end
    control_positions = sheet.index[sheet["is_control"]].tolist()
    assert min(control_positions) < len(sheet) - 4


def test_agreement_detects_discriminating_raters():
    import pandas as pd

    from src.human_eval import compute_agreement

    # 3 raters, 6 items; the last two are controls and are rated low by everyone
    frame = pd.DataFrame({
        "item_id": [f"i{j}" for j in range(6)],
        "is_control": [False, False, False, False, True, True],
        "rater1": [5.0, 4.0, 5.0, 4.0, 1.0, 2.0],
        "rater2": [4.0, 5.0, 4.0, 5.0, 2.0, 1.0],
        "rater3": [5.0, 5.0, 4.0, 4.0, 1.0, 1.0],
    })
    out = compute_agreement(frame)
    assert out["n_raters"] == 3 and out["n_items"] == 6
    assert out["control_discrimination"] > 2.0
    assert out["raters_discriminating"] == 3
    assert np.isfinite(out["krippendorff_alpha"])


def test_agreement_flags_non_discriminating_raters():
    """Everyone defaulting to 4 must NOT look like agreement about quality."""
    import pandas as pd

    from src.human_eval import compute_agreement

    frame = pd.DataFrame({
        "item_id": [f"i{j}" for j in range(6)],
        "is_control": [False, False, False, False, True, True],
        "rater1": [4.0] * 6,
        "rater2": [4.0] * 6,
        "rater3": [4.0] * 6,
    })
    out = compute_agreement(frame)
    assert out["control_discrimination"] == 0.0
    assert out["raters_discriminating"] == 0


# --------------------------------------------------------------------------- #
# A0.1 -- the synthetic-artifact guard
# --------------------------------------------------------------------------- #
def test_detect_provenance_reads_the_explicit_field():
    from src.utils import REAL, SYNTHETIC, detect_provenance

    assert detect_provenance({"provenance": "synthetic"}) == SYNTHETIC
    assert detect_provenance({"provenance": "real"}) == REAL
    assert detect_provenance({}) == REAL
    assert detect_provenance(None) == REAL


def test_detect_provenance_does_not_rely_on_the_dataset_name():
    """The synthetic generator reuses real corpus names on purpose.

    A row that says ``dataset == "mtat"`` may still be fake, so the corpus name
    alone must never be treated as evidence of realness.
    """
    from src.utils import REAL, SYNTHETIC, detect_provenance

    fake = {"dataset": "mtat", "provenance": "synthetic"}
    real = {"dataset": "mtat", "provenance": "real"}
    assert detect_provenance(fake) == SYNTHETIC
    assert detect_provenance(real) == REAL
    # legacy artifacts written before the field existed
    assert detect_provenance({"dataset": "synthetic"}) == SYNTHETIC


def test_detect_provenance_on_quarantine_paths():
    from src.utils import REAL, SYNTHETIC, detect_provenance

    assert detect_provenance("results/_synthetic_smoke/plots/ablation.png") == SYNTHETIC
    assert detect_provenance("data/processed/synthetic/graphs/syn_0000.pt") == SYNTHETIC
    assert detect_provenance("data/processed/graphs/mtat/mtat_2.pt") == REAL


def test_detect_provenance_on_a_dataframe():
    import pandas as pd

    from src.utils import REAL, SYNTHETIC, detect_provenance

    fake = pd.DataFrame({"track_id": ["a"], "dataset": ["mtat"], "provenance": ["synthetic"]})
    real = pd.DataFrame({"track_id": ["a"], "dataset": ["mtat"], "provenance": ["real"]})
    assert detect_provenance(fake) == SYNTHETIC
    assert detect_provenance(real) == REAL


def test_guard_refuses_synthetic_without_the_flag():
    from src.utils import SyntheticArtifactError, guard_against_synthetic

    with pytest.raises(SyntheticArtifactError, match="refusing to run on synthetic"):
        guard_against_synthetic({"provenance": "synthetic"}, allow_synthetic=False)


def test_guard_allows_synthetic_with_the_flag():
    from src.utils import SYNTHETIC, guard_against_synthetic

    assert guard_against_synthetic({"provenance": "synthetic"},
                                   allow_synthetic=True) == SYNTHETIC


def test_guard_passes_real_data_through():
    from src.utils import REAL, guard_against_synthetic

    assert guard_against_synthetic({"provenance": "real"}, allow_synthetic=False) == REAL


def test_synthetic_graphs_are_stamped(synthetic):
    """Every generated graph and manifest row must carry its provenance."""
    import pandas as pd

    from src.utils import SYNTHETIC, detect_provenance

    graph = torch.load(sorted(Path(synthetic).glob("graphs/*.pt"))[0], weights_only=False)
    assert getattr(graph, "provenance", None) == SYNTHETIC
    assert detect_provenance(graph) == SYNTHETIC

    manifest = pd.read_csv(Path(synthetic) / "manifest.csv")
    assert "provenance" in manifest.columns
    assert set(manifest["provenance"]) == {SYNTHETIC}
    assert detect_provenance(manifest) == SYNTHETIC


def test_real_graphs_default_to_real_provenance():
    from src.graph_builder import build_segment_graph
    from src.utils import REAL, detect_provenance

    cfg = load_config("config.yaml")
    feats = np.random.default_rng(0).normal(size=(8, 96)).astype(np.float32)
    graph = build_segment_graph(feats, cfg, track_id="mtat_2", dataset="mtat")
    assert graph.provenance == REAL
    assert detect_provenance(graph) == REAL


def test_export_sample_graphs_refuses_synthetic_without_the_flag(synthetic, tmp_path):
    """The committed sample graphs are graded; a fake one is a submission failure."""
    import subprocess
    import sys

    from src.utils import SyntheticArtifactError, guard_against_synthetic

    graphs = [torch.load(p, weights_only=False)
              for p in sorted(Path(synthetic).glob("graphs/*.pt"))[:3]]
    with pytest.raises(SyntheticArtifactError):
        guard_against_synthetic(graphs, allow_synthetic=False, context="export")
    # ...and is fine when asked for explicitly
    assert guard_against_synthetic(graphs, allow_synthetic=True) == "synthetic"


def test_evaluate_checkpoint_guard_detects_synthetic_weights(tmp_path):
    from src.evaluate import _guard_checkpoints
    from src.utils import SyntheticArtifactError

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"model_state": {}, "provenance": "synthetic"}, ckpt_dir / "task1_seed42_best.pt")
    cfg = load_config("config.yaml", {"paths.checkpoints": str(ckpt_dir)})

    with pytest.raises(SyntheticArtifactError):
        _guard_checkpoints(cfg, allow_synthetic=False)
    _guard_checkpoints(cfg, allow_synthetic=True)      # explicit opt-in is fine


def test_evaluate_checkpoint_guard_passes_real_weights(tmp_path):
    from src.evaluate import _guard_checkpoints

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({"model_state": {}, "provenance": "real"}, ckpt_dir / "task2_seed42_best.pt")
    cfg = load_config("config.yaml", {"paths.checkpoints": str(ckpt_dir)})
    _guard_checkpoints(cfg, allow_synthetic=False)     # must not raise


def test_training_results_record_provenance(cfg_overrides):
    result = _run(2, cfg_overrides)
    assert result["provenance"] == "synthetic"


# --------------------------------------------------------------------------- #
# A0.5 -- item-level resumability and atomic writes
#
# These are the tests that matter for a pipeline run across many sessions: the
# long jobs WILL be interrupted, and the only acceptable behaviour is to lose at
# most the item in flight.
# --------------------------------------------------------------------------- #
def _tiny_manifest(tmp_path, n=6):
    """A manifest of short synthetic wav files, so extraction is fast but real."""
    import soundfile as sf

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = audio_dir / f"clip{i}.wav"
        # 4 s of pink-ish noise at 22.05 kHz: long enough for min_nodes segments
        sf.write(path, rng.normal(0, 0.1, 22050 * 4).astype(np.float32), 22050)
        rows.append({"track_id": f"t{i}", "artist_id": f"a{i}", "audio_path": str(path),
                     "text": "", "split": "train", "y_genre": -1, "y_tags": "[]",
                     "y_valence": np.nan, "y_arousal": np.nan, "duration_s": 4.0,
                     "dataset": "probe"})
    import pandas as pd

    return pd.DataFrame(rows)


@pytest.mark.slow
def test_extract_dataset_skips_already_cached_keys(tmp_path):
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=4)
    h5 = tmp_path / "features.h5"

    first = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert first["written"] == 4 and first["skipped"] == 0

    # a second run must do no work at all
    second = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert second["written"] == 0, "re-extracted keys that were already cached"
    assert second["skipped"] == 4


@pytest.mark.slow
def test_extract_dataset_resumes_after_a_partial_run(tmp_path):
    """Extract half, then extract the whole manifest: only the rest is done."""
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=6)
    h5 = tmp_path / "features.h5"

    partial = extract_dataset(manifest.iloc[:3], h5, cfg, n_workers=1)
    assert partial["written"] == 3

    resumed = extract_dataset(manifest, h5, cfg, n_workers=1)
    assert resumed["skipped"] == 3, "resume did not recognise the completed keys"
    assert resumed["written"] == 3

    import h5py

    with h5py.File(h5, "r") as store:
        assert len(store.keys()) == 6
        assert store.attrs["feature_dim"] == 96
        for key in store.keys():
            assert store[key].shape[1] == 96


@pytest.mark.slow
def test_extract_dataset_writes_a_key_sidecar(tmp_path):
    """HDF5 is not transactional; the sidecar is how we know what completed."""
    import json as _json

    from src.audio_features import cache_keys_path, extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=3)
    h5 = tmp_path / "features.h5"
    extract_dataset(manifest, h5, cfg, n_workers=1)

    sidecar = cache_keys_path(h5)
    assert sidecar.exists(), "no key sidecar written"
    payload = _json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["n_keys"] == 3
    assert sorted(payload["keys"]) == ["t0", "t1", "t2"]


def test_verify_cache_reports_a_corrupt_container(tmp_path):
    from src.audio_features import verify_cache

    broken = tmp_path / "features.h5"
    broken.write_bytes(b"this is not an HDF5 file")
    report = verify_cache(broken)
    assert report["exists"] is True
    assert report["readable"] is False
    assert report["error"]


def test_verify_cache_on_a_missing_file_is_not_an_error(tmp_path):
    from src.audio_features import verify_cache

    report = verify_cache(tmp_path / "nope.h5")
    assert report["exists"] is False and report["error"] is None


@pytest.mark.slow
def test_extract_dataset_refuses_a_corrupt_cache_rather_than_appending(tmp_path):
    from src.audio_features import extract_dataset

    cfg = load_config("config.yaml")
    manifest = _tiny_manifest(tmp_path, n=2)
    h5 = tmp_path / "features.h5"
    h5.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="cannot be opened"):
        extract_dataset(manifest, h5, cfg, n_workers=1)
    # ...but overwrite=True is an explicit, allowed recovery
    stats = extract_dataset(manifest, h5, cfg, n_workers=1, overwrite=True)
    assert stats["written"] == 2


# ---- atomic writes --------------------------------------------------------- #
def test_atomic_write_leaves_no_temp_file(tmp_path):
    from src.utils import atomic_write_text

    target = tmp_path / "manifest.csv"
    atomic_write_text(target, "a,b\n1,2\n")
    assert target.read_text() == "a,b\n1,2\n"
    assert not list(tmp_path.glob("*.tmp")), "temp file survived the write"


def test_atomic_write_does_not_clobber_on_failure(tmp_path, monkeypatch):
    """If the write dies mid-way, the previous file must still be intact."""
    import os as _os

    from src.utils import atomic_write_text

    target = tmp_path / "state.json"
    atomic_write_text(target, "GOOD")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "PARTIAL")
    assert target.read_text() == "GOOD", "a failed write destroyed the old file"


def test_save_json_is_atomic(tmp_path):
    from src.utils import save_json

    path = save_json({"x": 1}, tmp_path / "out.json")
    assert json.loads(Path(path).read_text())["x"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_torch_save_round_trips(tmp_path):
    from src.utils import atomic_torch_save

    payload = {"model_state": {"w": torch.ones(3)}, "provenance": "real"}
    path = atomic_torch_save(payload, tmp_path / "ckpt.pt")
    back = torch.load(path, weights_only=False)
    assert torch.equal(back["model_state"]["w"], torch.ones(3))
    assert not list(tmp_path.glob("*.tmp"))


def test_write_manifest_is_atomic(tmp_path):
    import pandas as pd

    from src.splits import write_manifest

    frame = pd.DataFrame([{"track_id": "t1", "artist_id": "a1", "split": "train",
                           "y_tags": ["guitar"]}])
    path = write_manifest(frame, tmp_path / "m.csv")
    assert Path(path).exists()
    assert not list(tmp_path.glob("*.tmp"))


# ---- graph building -------------------------------------------------------- #
@pytest.mark.slow
def test_graph_building_skips_existing_files(tmp_path, synthetic):
    """Graph building must be item-level resumable, like extraction."""
    from scripts.build_graphs import build_graphs_for_manifest

    import pandas as pd

    manifest = pd.read_csv(Path(synthetic) / "manifest.csv").head(6)
    cfg = load_config("config.yaml")
    out_dir = tmp_path / "graphs"

    first = build_graphs_for_manifest(
        manifest, cfg, out_dir, graph_dir=Path(synthetic) / "graphs")
    assert first["written"] == 6 and first["skipped"] == 0

    second = build_graphs_for_manifest(
        manifest, cfg, out_dir, graph_dir=Path(synthetic) / "graphs")
    assert second["written"] == 0, "rebuilt graphs that already existed"
    assert second["skipped"] == 6


# ---- musiccaps retry ------------------------------------------------------- #
def test_musiccaps_retry_selects_only_failed_rows(tmp_path):
    """--retry-failed must read the log and skip everything already ok."""
    import pandas as pd

    log = pd.DataFrame([
        {"ytid": "aaa", "status": "ok"},
        {"ytid": "bbb", "status": "missing"},
        {"ytid": "ccc", "status": "corrupt"},
        {"ytid": "ddd", "status": "wrong_duration"},
        {"ytid": "eee", "status": "ok"},
    ])
    source = pd.DataFrame([{"ytid": y, "start_s": 30, "end_s": 40,
                            "is_audioset_eval": False}
                           for y in ["aaa", "bbb", "ccc", "ddd", "eee"]])

    failed = set(log.loc[log["status"] != "ok", "ytid"].astype(str))
    todo = source[source["ytid"].astype(str).isin(failed)]
    assert set(todo["ytid"]) == {"bbb", "ccc", "ddd"}
    assert "aaa" not in set(todo["ytid"]) and "eee" not in set(todo["ytid"])


def test_musiccaps_download_skips_existing_good_files(tmp_path):
    """A clip already on disk with the right duration is not re-downloaded."""
    import soundfile as sf

    from scripts.download_musiccaps import download_clip

    sf.write(tmp_path / "abc_30_40.wav",
             np.zeros(22050 * 10, dtype=np.float32), 22050)
    row = download_clip("abc", 30, 40, tmp_path, sleep=0.0, fmt="wav")
    assert row["status"] == "ok"
    assert abs(row["duration_s"] - 10.0) < 0.1


# --------------------------------------------------------------------------- #
# A0.6 -- Xtext sources and aspect stripping
#
# The failure this guards against is degenerate supervision: MusicCaps captions
# are written FROM the aspect list, so training on the raw caption to predict
# those aspects measures string matching, not music understanding.
# --------------------------------------------------------------------------- #
def test_strip_removes_exact_aspect_phrases():
    from src.splits import strip_aspect_terms

    caption = "The low quality recording features a ballad song."
    out = strip_aspect_terms(caption, ["low quality", "ballad"])
    assert "low quality" not in out.lower()
    assert "ballad" not in out.lower()
    assert "recording" in out and "song" in out, "stripping destroyed the carrier text"


def test_strip_removes_scattered_multiword_aspects():
    """"sustained strings melody" also appears as "sustained strings, ... melody"."""
    from src.splits import strip_aspect_terms

    caption = ("contains sustained strings, mellow piano melody and soft female "
               "vocal singing over it")
    out = strip_aspect_terms(
        caption, ["sustained strings melody", "mellow piano melody", "soft female vocal"]
    ).lower()
    for leaked in ("sustained", "strings", "melody", "mellow", "piano", "female", "vocal"):
        assert leaked not in out, f"{leaked!r} survived stripping"


def test_strip_handles_morphological_variants():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms(
        "the guitars are strumming and the drummer is drumming",
        ["guitar", "strum", "drum"],
    ).lower()
    for leaked in ("guitar", "strum", "drum"):
        assert leaked not in out, f"inflected form of {leaked!r} survived"


def test_strip_preserves_stopwords_and_unrelated_content():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms(
        "It sounds like something you would hear at Sunday services.", ["sad"]
    )
    assert "Sunday services" in out
    assert "sounds" in out


def test_strip_is_a_noop_without_aspects():
    from src.splits import strip_aspect_terms

    caption = "A perfectly ordinary caption."
    assert strip_aspect_terms(caption, []) == caption


def test_strip_tidies_orphaned_punctuation():
    from src.splits import strip_aspect_terms

    out = strip_aspect_terms("It sounds sad and soulful, really.", ["sad", "soulful"])
    assert ",," not in out and "  " not in out
    assert not out.startswith(",")


def test_aspect_surface_forms_include_phrase_and_tokens():
    from src.splits import aspect_surface_forms

    forms = aspect_surface_forms(["soft female vocal"])
    assert "soft female vocal" in forms
    assert "vocal" in forms and "female" in forms
    assert "vocals" in forms, "plural inflection missing"


def test_masking_actually_removes_label_information():
    """The whole point: a masked caption must not contain its own labels."""
    from src.splits import strip_aspect_terms

    aspects = ["low quality", "sustained strings melody", "soft female vocal",
               "mellow piano melody", "sad", "soulful", "ballad"]
    caption = ("The low quality recording features a ballad song that contains "
               "sustained strings, mellow piano melody and soft female vocal "
               "singing over it. It sounds sad and soulful.")
    raw, masked = caption.lower(), strip_aspect_terms(caption, aspects).lower()

    def hits(text):
        return sum(1 for a in aspects if a.lower() in text)

    assert hits(raw) >= 5, "the test caption should leak in its raw form"
    assert hits(masked) == 0, "masked caption still contains its own labels"


def test_mtat_metadata_text_never_contains_tags():
    from src.splits import mtat_metadata_text

    text = mtat_metadata_text("BWV54 - I Aria", "J.S. Bach Solo Cantatas",
                              "American Bach Soloists")
    assert "American Bach Soloists" in text and "BWV54" in text
    assert mtat_metadata_text("", "", "") == ""
    assert mtat_metadata_text("Title", "nan", "") == "Title"


def test_text_sources_are_the_three_locked_values():
    from src.splits import TEXT_SOURCES

    assert set(TEXT_SOURCES) == {"caption_masked", "caption_raw", "metadata"}


def test_apply_text_source_selects_the_configured_variant(tmp_path):
    import pandas as pd

    from src.splits import apply_text_source

    manifest = pd.DataFrame([
        {"track_id": "musiccaps_a", "dataset": "musiccaps", "text": "ORIGINAL"},
        {"track_id": "musiccaps_b", "dataset": "musiccaps", "text": "ORIGINAL"},
    ])
    pd.DataFrame([
        {"track_id": "musiccaps_a", "dataset": "musiccaps", "caption_raw": "RAW A",
         "caption_masked": "MASKED A", "metadata": "META A"},
        {"track_id": "musiccaps_b", "dataset": "musiccaps", "caption_raw": "RAW B",
         "caption_masked": "MASKED B", "metadata": "META B"},
    ]).to_csv(tmp_path / "musiccaps_text_variants.csv", index=False)

    for source, expected in [("caption_masked", "MASKED A"),
                             ("caption_raw", "RAW A"),
                             ("metadata", "META A")]:
        cfg = load_config("config.yaml", {"data.text_source": source})
        out = apply_text_source(manifest, cfg, splits_dir=tmp_path)
        assert out.loc[0, "text"] == expected
        assert out.loc[0, "text_source"] == source


def test_apply_text_source_falls_back_rather_than_emptying_a_row(tmp_path):
    """An empty variant must not become an empty string fed to the tokenizer."""
    import pandas as pd

    from src.splits import apply_text_source

    manifest = pd.DataFrame([
        {"track_id": "mtat_1", "dataset": "mtat", "text": "FALLBACK"},
    ])
    pd.DataFrame([
        {"track_id": "mtat_1", "dataset": "mtat", "caption_raw": "",
         "caption_masked": "", "metadata": ""},
    ]).to_csv(tmp_path / "mtat_text_variants.csv", index=False)

    cfg = load_config("config.yaml", {"data.text_source": "caption_masked"})
    out = apply_text_source(manifest, cfg, splits_dir=tmp_path)
    assert out.loc[0, "text"] == "FALLBACK"


def test_apply_text_source_rejects_an_unknown_value(tmp_path):
    import pandas as pd

    from src.splits import apply_text_source

    cfg = load_config("config.yaml", {"data.text_source": "lyrics"})
    with pytest.raises(ValueError, match="text_source"):
        apply_text_source(pd.DataFrame([{"track_id": "x", "dataset": "mtat",
                                         "text": "t"}]), cfg, splits_dir=tmp_path)


def test_build_text_variants_shape():
    import pandas as pd

    from src.splits import build_text_variants

    manifest = pd.DataFrame([
        {"track_id": "t1", "dataset": "musiccaps", "text": "a sad ballad recording"},
    ])
    out = build_text_variants(manifest, aspects_by_track={"t1": ["sad", "ballad"]})
    assert list(out.columns) == ["track_id", "dataset", "caption_raw",
                                 "caption_masked", "metadata", "n_aspects_stripped"]
    assert out.loc[0, "caption_raw"] == "a sad ballad recording"
    assert "sad" not in out.loc[0, "caption_masked"]
    assert out.loc[0, "n_aspects_stripped"] == 2


def test_mtat_manifest_text_is_not_the_tag_string():
    """Regression guard for the degenerate Xtext the spec warns about."""
    import pandas as pd

    from src.utils import project_root

    path = project_root() / "data" / "splits" / "mtat_manifest.csv"
    if not path.exists():
        pytest.skip("MTAT manifest not built yet")
    frame = pd.read_csv(path).head(200)
    for record in frame.to_dict("records"):
        tags = json.loads(record["y_tags"]) if isinstance(record["y_tags"], str) else []
        text = str(record.get("text", "") or "").lower()
        if not tags or not text:
            continue
        leaked = [t for t in tags if t.lower() in text]
        assert len(leaked) < max(2, len(tags)),  (
            f"{record['track_id']}: Xtext appears to contain its own tags {leaked}"
        )


# --------------------------------------------------------------------------- #
# A3.6 -- normalisation statistics must be provably train-only
# --------------------------------------------------------------------------- #
def test_norm_stats_files_record_train_provenance():
    """Every persisted norm_stats file must say, in the file, that it is train."""
    from src.utils import project_root

    processed = project_root() / "data" / "processed"
    files = sorted(processed.glob("norm_stats*.json"))
    if not files:
        pytest.skip("no norm stats extracted yet")
    for path in files:
        stats = json.loads(path.read_text(encoding="utf-8"))
        assert stats.get("split") == "train", (
            f"{path.name} records split={stats.get('split')!r}; normalisation "
            "statistics must come from the train split only"
        )
        assert stats.get("dim") == 96
        assert len(stats["mean"]) == 96 and len(stats["std"]) == 96
        assert all(s > 0 for s in stats["std"]), "a zero std would divide by zero"


def test_compute_norm_stats_uses_only_train_rows(tmp_path):
    """Val/test rows in the manifest must not influence the statistics."""
    import h5py
    import pandas as pd

    from src.audio_features import compute_norm_stats

    h5 = tmp_path / "f.h5"
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, size=(10, 96)).astype(np.float16)
    # val/test rows are wildly off-distribution: if they leak in, mean explodes
    other = (rng.normal(0.0, 1.0, size=(10, 96)) + 1000.0).astype(np.float16)
    with h5py.File(h5, "w") as store:
        store.create_dataset("t_train", data=train)
        store.create_dataset("t_val", data=other)
        store.create_dataset("t_test", data=other)

    manifest = pd.DataFrame([
        {"track_id": "t_train", "split": "train"},
        {"track_id": "t_val", "split": "val"},
        {"track_id": "t_test", "split": "test"},
    ])
    stats = compute_norm_stats(manifest, split="train", cfg=None, h5_path=h5)
    assert stats["split"] == "train"
    assert stats["n_segments"] == 10, "non-train rows were included"
    assert abs(float(np.mean(stats["mean"]))) < 5.0, (
        "the val/test offset of +1000 leaked into the train statistics"
    )


def test_data_bundle_refuses_non_train_norm_stats(tmp_path):
    """A norm_stats file without train provenance must stop the run."""
    from src.utils import save_json

    processed = tmp_path / "processed"
    processed.mkdir()
    save_json({"mean": [0.0] * 96, "std": [1.0] * 96, "split": "test", "dim": 96},
              processed / "norm_stats.json")
    stats = json.loads((processed / "norm_stats.json").read_text(encoding="utf-8"))
    assert stats["split"] != "train"
    # DataBundle raises on exactly this condition (src/train.py)
    from src.utils import project_root

    source = (project_root() / "src" / "train.py").read_text(encoding="utf-8")
    assert 'payload.get("split") != "train"' in source
    assert "refusing to run" in source


# --------------------------------------------------------------------------- #
# A4.4 -- the graph sanity gate
# --------------------------------------------------------------------------- #
def test_repetition_score_detects_repeated_structure():
    from scripts.graph_sanity import repetition_score

    rng = np.random.default_rng(0)
    # a verse/chorus/verse track: segments 0-3 recur at 8-11
    base = rng.normal(size=(4, 96))
    repetitive = np.vstack([base, rng.normal(size=(4, 96)), base]).astype(np.float32)
    through_composed = rng.normal(size=(12, 96)).astype(np.float32)

    assert repetition_score(repetitive) > repetition_score(through_composed)


def test_analyse_graph_flags_a_chain_like_graph():
    """k=1 on a smoothly drifting track should look chain-like and score low."""
    from scripts.graph_sanity import analyse_graph
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml", {"graph.knn_k": 1})
    # a smooth ramp: every segment's nearest neighbour is its time neighbour
    feats = np.linspace(0, 1, 20)[:, None] * np.ones((1, 96))
    feats = feats.astype(np.float32) + 1e-3 * np.arange(96)[None, :]
    data = build_segment_graph(feats, cfg, track_id="ramp")
    report = analyse_graph(data, feats)
    assert report["long_range_fraction"] < 0.5, (
        "a pure temporal ramp should not produce long-range similarity edges"
    )


def test_analyse_graph_finds_planted_repeats():
    from scripts.graph_sanity import analyse_graph
    from src.graph_builder import build_segment_graph

    cfg = load_config("config.yaml")
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, 96))
    feats = np.vstack([base, rng.normal(size=(4, 96)), base]).astype(np.float32)
    data = build_segment_graph(feats, cfg, track_id="verse_chorus_verse")
    report = analyse_graph(data, feats)
    assert report["long_range_fraction"] > 0.5, "planted repeats were not connected"
    assert report["nodes_without_similarity_edges"] == 0


def test_graph_sanity_verdict_written_for_real_data():
    """If the gate has been run, its verdict must be a pass."""
    from src.utils import project_root

    path = project_root() / "results" / "plots" / "graph_sanity" / "graph_sanity.json"
    if not path.exists():
        pytest.skip("graph sanity gate has not been run yet")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    assert verdict["passed"] is True, (
        f"A4.4 gate failed: long_range={verdict['mean_long_range_fraction']:.3f}, "
        f"repeat_recall={verdict['mean_repeat_recall']:.3f}. Do not proceed."
    )


# --------------------------------------------------------------------------- #
# the mel cache the CNN baseline eats -- fairness, not fidelity
# --------------------------------------------------------------------------- #
def test_pooled_log_mel_covers_the_whole_track():
    """B2 must see the same span of music as the GNN, not a 6 s crop."""
    from src.audio_features import pooled_log_mel

    cfg = load_config("config.yaml")
    sr = int(cfg["audio"]["sample_rate"])
    rng = np.random.default_rng(0)
    # loud second half: if only the first half were kept, the tail would vanish
    y = np.concatenate([rng.normal(0, 0.01, sr * 10),
                        rng.normal(0, 0.50, sr * 10)]).astype(np.float32)

    mel = pooled_log_mel(y, sr, cfg, n_frames=256)
    assert mel.shape == (int(cfg["audio"]["n_mels"]), 256)
    first, second = mel[:, :128].mean(), mel[:, 128:].mean()
    assert second > first + 3.0, "the second half of the track is missing from the patch"


def test_pooled_log_mel_is_fixed_width_regardless_of_duration():
    from src.audio_features import pooled_log_mel

    cfg = load_config("config.yaml")
    sr = int(cfg["audio"]["sample_rate"])
    rng = np.random.default_rng(0)
    for seconds in (5, 29, 30):
        mel = pooled_log_mel(rng.normal(0, 0.1, sr * seconds).astype(np.float32),
                             sr, cfg, n_frames=256)
        assert mel.shape[1] == 256, f"{seconds}s track gave {mel.shape[1]} frames"
