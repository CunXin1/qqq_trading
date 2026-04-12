"""Tests for feature registry — single source of truth."""
from qqq_trading.features.registry import (
    get_base_features, get_refined_external_features,
    get_interaction_features, get_path_features,
    get_full_features, get_0dte_premarket_features,
)


def test_no_duplicates_in_base():
    feats = get_base_features()
    assert len(feats) == len(set(feats))


def test_no_duplicates_in_external():
    feats = get_refined_external_features()
    assert len(feats) == len(set(feats))


def test_no_duplicates_in_interactions():
    feats = get_interaction_features()
    assert len(feats) == len(set(feats))


def test_no_duplicates_in_path():
    feats = get_path_features()
    assert len(feats) == len(set(feats))


def test_no_overlap_between_groups():
    base = set(get_base_features())
    ext = set(get_refined_external_features())
    ix = set(get_interaction_features())
    path = set(get_path_features())

    assert not base & ext, f"Overlap base/ext: {base & ext}"
    assert not base & ix, f"Overlap base/ix: {base & ix}"
    assert not ext & path, f"Overlap ext/path: {ext & path}"


def test_base_feature_count():
    assert len(get_base_features()) == 53


def test_full_features_includes_base():
    full = get_full_features()
    base = get_base_features()
    for f in base:
        assert f in full


def test_full_features_composition():
    full = get_full_features(include_interactions=True, include_path=False)
    assert len(full) == len(get_base_features()) + len(get_refined_external_features()) + len(get_interaction_features())


def test_premarket_features():
    pre = get_0dte_premarket_features()
    assert "premarket_ret_today" in pre
    assert len(pre) == 3
