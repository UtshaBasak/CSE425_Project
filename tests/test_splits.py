"""The leakage assertion and the tag-vocabulary reduction.

``assert_no_leakage`` runs at the top of every training script, so it has to be
strict in both directions: it must raise on a genuinely leaked manifest, and it
must not raise on a clean one (a false positive would block every run).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    MTAT_SYNONYMS,
    assert_no_leakage,
    load_manifest,
    reduce_to_top_k_tags,
    write_manifest,
)


def _manifest(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.fixture
def clean():
    return _manifest([
        {"track_id": "t1", "artist_id": "a1", "split": "train"},
        {"track_id": "t2", "artist_id": "a1", "split": "train"},
        {"track_id": "t3", "artist_id": "a2", "split": "val"},
        {"track_id": "t4", "artist_id": "a3", "split": "test"},
    ])


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_clean_manifest_passes(clean):
    assert_no_leakage(clean)          # must not raise


def test_track_id_in_two_splits_raises(clean):
    leaked = pd.concat([clean, _manifest([
        {"track_id": "t1", "artist_id": "a1", "split": "test"},
    ])], ignore_index=True)
    with pytest.raises(AssertionError, match="track_id"):
        assert_no_leakage(leaked)


def test_artist_in_two_splits_raises(clean):
    leaked = pd.concat([clean, _manifest([
        {"track_id": "t5", "artist_id": "a1", "split": "test"},
    ])], ignore_index=True)
    with pytest.raises(AssertionError, match="artist_id"):
        assert_no_leakage(leaked)


def test_artist_leak_is_caught_even_with_distinct_track_ids():
    """The subtle case: no repeated track, but the same artist on both sides."""
    leaked = _manifest([
        {"track_id": "x1", "artist_id": "the_band", "split": "train"},
        {"track_id": "x2", "artist_id": "the_band", "split": "test"},
    ])
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked)


def test_blank_artist_ids_are_singletons_not_one_group():
    """Missing artist metadata must not fabricate a leak across every split."""
    frame = _manifest([
        {"track_id": "t1", "artist_id": "", "split": "train"},
        {"track_id": "t2", "artist_id": np.nan, "split": "val"},
        {"track_id": "t3", "artist_id": "unknown", "split": "test"},
    ])
    assert_no_leakage(frame)


def test_empty_manifest_is_not_an_error():
    assert_no_leakage(pd.DataFrame(columns=["track_id", "artist_id", "split"]))


def test_missing_required_column_raises():
    with pytest.raises(KeyError):
        assert_no_leakage(pd.DataFrame({"artist_id": ["a"], "split": ["train"]}))


def test_accepts_a_csv_path(tmp_path, clean):
    path = tmp_path / "m.csv"
    clean.to_csv(path, index=False)
    assert_no_leakage(str(path))


# --------------------------------------------------------------------------- #
# tag vocabulary
# --------------------------------------------------------------------------- #
def _annotations():
    # 'vocal' and 'vocals' are the canonical MTAT duplicate pair
    return pd.DataFrame({
        "clip_id": [1, 2, 3, 4, 5],
        "vocal":   [1, 1, 0, 0, 0],
        "vocals":  [0, 0, 1, 1, 0],
        "guitar":  [1, 0, 1, 0, 1],
        "sitar":   [0, 0, 0, 0, 1],
        "mp3_path": ["a/1.mp3", "b/2.mp3", "c/3.mp3", "d/4.mp3", "e/5.mp3"],
    })


def test_synonyms_merge_before_counting():
    reduced, tags = reduce_to_top_k_tags(_annotations(), k=2, merge_synonyms=True)
    assert "vocals" not in tags, "the variant must be folded into its canonical form"
    assert "vocal" in tags
    # merged 'vocal' now covers 4 clips, so it outranks guitar (3)
    assert int(reduced["vocal"].sum()) == 4
    assert tags[0] == "vocal"


def test_without_merging_the_duplicates_split_the_support():
    reduced, tags = reduce_to_top_k_tags(_annotations(), k=4, merge_synonyms=False)
    assert int(reduced["vocal"].sum()) == 2
    assert int(reduced["vocals"].sum()) == 2


def test_top_k_keeps_exactly_k_tags():
    _, tags = reduce_to_top_k_tags(_annotations(), k=2, merge_synonyms=True)
    assert len(tags) == 2


def test_known_mtat_duplicate_pairs_are_covered():
    """The pairs the spec calls out by name must be in the merge table."""
    flat = {v for variants in MTAT_SYNONYMS.values() for v in variants}
    flat |= set(MTAT_SYNONYMS)
    for pair in ("vocal", "vocals", "choir", "choral", "beat", "beats",
                 "female vocal", "female vocals"):
        assert pair in flat, f"{pair!r} missing from the synonym table"


def test_mp3_path_column_is_not_treated_as_a_tag():
    _, tags = reduce_to_top_k_tags(_annotations(), k=10, merge_synonyms=True)
    assert "mp3_path" not in tags
    assert "clip_id" not in tags


# --------------------------------------------------------------------------- #
# manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_round_trip_preserves_tag_lists(tmp_path):
    frame = pd.DataFrame([{
        "track_id": "t1", "artist_id": "a1", "audio_path": "x.mp3",
        "text": "guitar, drum", "split": "train", "y_genre": -1,
        "y_tags": ["guitar", "drum"], "y_valence": np.nan,
        "y_arousal": np.nan, "duration_s": 29.0,
    }])
    path = write_manifest(frame, tmp_path / "m.csv")
    back = load_manifest(path)
    assert back.loc[0, "y_tags"] == ["guitar", "drum"]
    assert np.isnan(back.loc[0, "y_valence"])


def test_written_manifest_has_the_contract_columns(tmp_path):
    frame = pd.DataFrame([{"track_id": "t1", "artist_id": "a1", "split": "train",
                           "y_tags": []}])
    path = write_manifest(frame, tmp_path / "m.csv")
    columns = list(pd.read_csv(path).columns)
    assert columns == ["track_id", "artist_id", "audio_path", "text", "split",
                       "y_genre", "y_tags", "y_valence", "y_arousal", "duration_s"]
