"""Structural guarantees of the graph builder, per the frozen data contract."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.audio_features import NODE_FEAT_DIM
from src.chords import CHORD_VOCAB
from src.graph_builder import (
    build_chord_graph,
    build_hetero_graph,
    build_segment_graph,
    knn_edge_index,
    rewire_edges,
)
from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config("config.yaml")


@pytest.fixture
def feats():
    rng = np.random.default_rng(0)
    return rng.normal(size=(12, NODE_FEAT_DIM)).astype(np.float32)


@pytest.fixture
def graph(feats, cfg):
    return build_segment_graph(feats, cfg, track_id="t0", artist_id="a0",
                               dataset="mtat", split="train", text="a test track",
                               y_tags=np.array([1.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
def test_node_feature_dim_is_exactly_96(graph, cfg):
    assert NODE_FEAT_DIM == 96
    assert graph.x.shape[1] == 96 == int(cfg["graph"]["node_feat_dim"])
    assert graph.x.dtype == torch.float32


def test_wrong_feature_dim_raises(cfg):
    with pytest.raises(AssertionError, match="node feature dim"):
        build_segment_graph(np.zeros((5, 64), dtype=np.float32), cfg)


def test_edge_index_shape_and_dtype(graph):
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.dtype == torch.int64


def test_edge_attr_shape_is_num_edges_by_two(graph):
    assert graph.edge_attr.shape == (graph.edge_index.shape[1], 2)
    assert graph.edge_attr.dtype == torch.float32


def test_edge_attr_channels_are_flag_then_cosine(graph):
    flags = graph.edge_attr[:, 0]
    assert torch.all((flags == 0) | (flags == 1))
    cos = graph.edge_attr[:, 1]
    assert torch.all(cos >= -1.0001) and torch.all(cos <= 1.0001)


def test_label_sentinels_when_labels_absent(feats, cfg):
    data = build_segment_graph(feats, cfg, n_tags=5)
    assert torch.all(data.y_tags == -1)
    assert int(data.y_genre.item()) == -1
    assert torch.isnan(data.y_valence).all()
    assert torch.isnan(data.y_arousal).all()


def test_tags_are_row_shaped_for_batching(graph):
    assert graph.y_tags.dim() == 2 and graph.y_tags.shape[0] == 1


def test_contract_string_fields_present(graph):
    for field in ("track_id", "artist_id", "dataset", "split", "text"):
        assert isinstance(getattr(graph, field), str)


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #
def test_knn_gives_constant_out_degree(feats, cfg):
    k = int(cfg["graph"]["knn_k"])
    edges = knn_edge_index(feats, k)
    counts = np.bincount(edges[0], minlength=feats.shape[0])
    assert edges.shape == (2, feats.shape[0] * k)
    assert np.all(counts == k), f"degrees vary: {counts}"


def test_knn_never_connects_a_node_to_itself(feats, cfg):
    edges = knn_edge_index(feats, int(cfg["graph"]["knn_k"]))
    assert not np.any(edges[0] == edges[1])


def test_knn_clamps_k_on_tiny_graphs(cfg):
    small = np.random.default_rng(0).normal(size=(3, NODE_FEAT_DIM)).astype(np.float32)
    edges = knn_edge_index(small, 10)
    assert edges.shape[1] == 3 * 2          # min(k, n-1) == 2 per node


def test_no_isolated_nodes(graph):
    n = int(graph.num_nodes)
    touched = set(graph.edge_index[0].tolist()) | set(graph.edge_index[1].tolist())
    assert touched == set(range(n))


def test_no_isolated_nodes_without_similarity_edges(feats, cfg):
    local = load_config("config.yaml", {"graph.similarity_edges": False})
    data = build_segment_graph(feats, local)
    non_self = data.edge_index[0] != data.edge_index[1]
    degrees = np.bincount(data.edge_index[0][non_self].numpy(),
                          minlength=int(data.num_nodes))
    assert np.all(degrees > 0), "temporal edges alone must still connect every node"


def test_self_loops_present(graph):
    self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
    assert self_loops == int(graph.num_nodes)


def test_self_loops_absent_when_disabled(feats):
    local = load_config("config.yaml", {"graph.add_self_loops": False})
    data = build_segment_graph(feats, local)
    assert (data.edge_index[0] == data.edge_index[1]).sum().item() == 0


def test_edges_are_bidirectional(graph):
    pairs = {(int(a), int(b)) for a, b in zip(*graph.edge_index)}
    missing = [(a, b) for (a, b) in pairs if a != b and (b, a) not in pairs]
    assert not missing, f"one-way edges found: {missing[:5]}"


def test_temporal_chain_is_present(graph):
    pairs = {(int(a), int(b)) for a, b in zip(*graph.edge_index)}
    for i in range(int(graph.num_nodes) - 1):
        assert (i, i + 1) in pairs and (i + 1, i) in pairs


def test_max_nodes_is_respected(cfg):
    big = np.random.default_rng(0).normal(size=(200, NODE_FEAT_DIM)).astype(np.float32)
    data = build_segment_graph(big, cfg)
    assert int(data.num_nodes) == int(cfg["segmentation"]["max_nodes"])


def test_similar_segments_become_neighbours(cfg):
    """A repeated section must be linked by a similarity edge, not just by time."""
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(10, NODE_FEAT_DIM)).astype(np.float32) * 0.1
    feats[7] = feats[1] + rng.normal(0, 1e-4, NODE_FEAT_DIM).astype(np.float32)
    data = build_segment_graph(feats, cfg)
    pairs = {(int(a), int(b)) for a, b in zip(*data.edge_index)}
    assert (1, 7) in pairs and (7, 1) in pairs


# --------------------------------------------------------------------------- #
# the rewiring control
# --------------------------------------------------------------------------- #
def _degrees(data):
    n = int(data.num_nodes)
    non_self = data.edge_index[0] != data.edge_index[1]
    return np.bincount(data.edge_index[0][non_self].numpy(), minlength=n)


def test_rewiring_preserves_degree(graph):
    rewired = rewire_edges(graph, preserve_degree=True, seed=7)
    assert np.array_equal(_degrees(graph), _degrees(rewired))


def test_rewiring_preserves_edge_count_and_node_count(graph):
    rewired = rewire_edges(graph, preserve_degree=True, seed=7)
    assert rewired.edge_index.shape[1] == graph.edge_index.shape[1]
    assert int(rewired.num_nodes) == int(graph.num_nodes)


def test_rewiring_actually_changes_the_topology(graph):
    rewired = rewire_edges(graph, preserve_degree=True, seed=7)
    before = {(int(a), int(b)) for a, b in zip(*graph.edge_index)}
    after = {(int(a), int(b)) for a, b in zip(*rewired.edge_index)}
    assert before != after


def test_rewiring_keeps_features_and_labels(graph):
    rewired = rewire_edges(graph, preserve_degree=True, seed=7)
    assert torch.equal(rewired.x, graph.x)
    assert rewired.track_id == graph.track_id
    assert torch.equal(rewired.y_tags, graph.y_tags)


def test_rewired_edges_are_marked_non_temporal(graph):
    rewired = rewire_edges(graph, preserve_degree=True, seed=7)
    assert torch.all(rewired.edge_attr[:, 0] == 0)


# --------------------------------------------------------------------------- #
# chord and hetero graphs
# --------------------------------------------------------------------------- #
def test_chord_graph_within_vocabulary(cfg):
    sequence = ["C:maj", "G:maj", "A:min", "F:maj", "C:maj", "N"]
    chroma = np.abs(np.random.default_rng(0).normal(size=(12, len(sequence))))
    data = build_chord_graph(sequence, chroma, cfg, track_id="t0")
    assert int(data.num_nodes) <= int(cfg["graph"]["chord_vocab_size"]) == len(CHORD_VOCAB)
    assert data.x.shape[1] == int(cfg["graph"]["chord_feat_dim"])
    assert data.edge_attr.shape[1] == 1


def test_chord_transition_weights_are_row_normalised(cfg):
    sequence = ["C:maj", "G:maj", "C:maj", "G:maj", "A:min"]
    data = build_chord_graph(sequence, None, cfg)
    src = data.edge_index[0].numpy()
    weights = data.edge_attr[:, 0].numpy()
    for node in np.unique(src):
        assert weights[src == node].sum() == pytest.approx(1.0, abs=1e-5)


def test_hetero_graph_has_all_five_edge_types(feats, cfg):
    seg = build_segment_graph(feats, cfg, track_id="t0")
    chords = ["C:maj", "G:maj", "A:min"]
    chord_graph = build_chord_graph(chords, None, cfg)
    seg2chord = [i % chord_graph.num_nodes for i in range(int(seg.num_nodes))]
    het = build_hetero_graph(seg, chord_graph, seg2chord)
    for edge_type in (("segment", "next", "segment"),
                      ("segment", "similar", "segment"),
                      ("segment", "contains", "chord"),
                      ("chord", "in", "segment"),
                      ("chord", "transitions", "chord")):
        assert edge_type in het.edge_types
    assert het["segment"].num_nodes == int(seg.num_nodes)
    assert het["chord"].num_nodes == int(chord_graph.num_nodes)


def test_hetero_contains_and_in_are_transposes(feats, cfg):
    seg = build_segment_graph(feats, cfg)
    chord_graph = build_chord_graph(["C:maj", "G:maj"], None, cfg)
    seg2chord = [0] * int(seg.num_nodes)
    het = build_hetero_graph(seg, chord_graph, seg2chord)
    forward = het["segment", "contains", "chord"].edge_index
    backward = het["chord", "in", "segment"].edge_index
    assert torch.equal(forward.flip(0), backward)
