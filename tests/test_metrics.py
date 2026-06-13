"""Metric library: known inputs -> known scores, and the name registry."""

import math

import pytest

from eval_lib import metrics


def test_accuracy_perfect_and_known():
    assert metrics.get("accuracy")([1, 0, 1], [1, 0, 1]) == 1.0
    assert metrics.get("accuracy")([1, 0, 0], [1, 0, 1]) == pytest.approx(2 / 3)


def test_f1_binary_known():
    # preds=[1,1,0,0] refs=[1,0,0,1]: tp=1, fp=1, fn=1 -> f1 = 0.5
    assert metrics.get("f1")([1, 1, 0, 0], [1, 0, 0, 1]) == pytest.approx(0.5)


def test_f1_macro_multiclass():
    score = metrics.get("f1_macro")([0, 1, 2, 0], [0, 1, 2, 1])
    assert 0.0 <= score <= 1.0 and not math.isnan(score)


def test_evaluate_runs_all_named():
    out = metrics.evaluate([1, 0], [1, 0], ["accuracy", "f1"])
    assert set(out) == {"accuracy", "f1"} and out["accuracy"] == 1.0


def test_unknown_metric_rejected():
    with pytest.raises(KeyError):
        metrics.get("bleu_imaginary")
