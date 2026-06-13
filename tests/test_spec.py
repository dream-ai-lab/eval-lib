"""Spec validation + canonical hash — the standard's enforcement layer."""

import pytest

from eval_lib.spec import SpecError, canonical_hash, validate


def _valid_raw():
    return {
        "paper_id": "x",
        "dataset": {
            "hf_id": "org/ds",
            "split": "test",
            "version": "1.0.0",
            "revision": "abc123",
            "text_field": "text",
            "label_field": "label",
        },
        "model": {"hf_id": "org/m", "revision": "def456"},
        "metrics": {"primary": "accuracy", "secondary": ["f1"]},
        "reproduce_target": {"accuracy": {"min": 0.9, "max": 0.92}},
    }


def test_valid_spec_passes():
    validate(_valid_raw())


def test_unpinned_dataset_rejected():
    raw = _valid_raw()
    raw["dataset"]["version"] = "latest"
    with pytest.raises(SpecError):
        validate(raw)


def test_unpinned_model_revision_rejected():
    raw = _valid_raw()
    raw["model"]["revision"] = "main"
    with pytest.raises(SpecError):
        validate(raw)


def test_unknown_metric_rejected():
    raw = _valid_raw()
    raw["metrics"]["primary"] = "not_a_metric"
    with pytest.raises(SpecError):
        validate(raw)


def test_canonical_hash_is_order_independent():
    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"b": {"d": 3, "c": 2}, "a": 1}  # reordered
    assert canonical_hash(a) == canonical_hash(b)
