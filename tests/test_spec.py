"""Spec validation + canonical hash — the standard's enforcement layer."""

import pytest

from eval_lib.spec import (
    SpecError,
    canonical_hash,
    check_experimental_provided,
    validate,
)


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


def test_experimental_metric_allowed():
    raw = _valid_raw()
    raw["metrics"] = {"primary": "mcc", "experimental": ["mcc"]}
    raw["reproduce_target"] = {"mcc": {"min": 0.5, "max": 0.6}}
    validate(raw)  # not in registry, but declared experimental -> ok


def test_experimental_cannot_shadow_standard_metric():
    raw = _valid_raw()
    raw["metrics"]["experimental"] = ["accuracy"]  # accuracy is a registry metric
    with pytest.raises(SpecError):
        validate(raw)


def test_unknown_metric_still_rejected_without_experimental():
    raw = _valid_raw()
    raw["metrics"]["secondary"] = ["mcc"]  # not registered, not experimental
    with pytest.raises(SpecError):
        validate(raw)


def test_check_experimental_provided_missing():
    with pytest.raises(SpecError):
        check_experimental_provided(["mcc"], provided=[])


def test_check_experimental_provided_stray():
    with pytest.raises(SpecError):
        check_experimental_provided([], provided=["mcc"])


def test_check_experimental_provided_match():
    check_experimental_provided(["mcc"], provided=["mcc"])  # no raise


def test_canonical_hash_is_order_independent():
    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"b": {"d": 3, "c": 2}, "a": 1}  # reordered
    assert canonical_hash(a) == canonical_hash(b)
