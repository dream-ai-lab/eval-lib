"""Shared metric library — the ONLY place metrics are implemented.

Every experiment imports metrics from here by NAME (referenced in
``eval_spec.yaml``). Nobody writes their own eval function. This is what
makes numbers comparable across people and across papers.

To add a metric: write a function ``(preds, refs) -> float`` and decorate
it with ``@register("name")``. Then it is usable from any eval_spec by
that name, and CI will accept specs that reference it.
"""

from __future__ import annotations

from typing import Callable, Sequence

from sklearn.metrics import accuracy_score, f1_score

_REGISTRY: dict[str, Callable[[Sequence, Sequence], float]] = {}


def register(name: str):
    def deco(fn: Callable[[Sequence, Sequence], float]):
        if name in _REGISTRY:
            raise ValueError(f"metric '{name}' already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


@register("accuracy")
def accuracy(preds: Sequence, refs: Sequence) -> float:
    return float(accuracy_score(refs, preds))


@register("f1")
def f1_binary(preds: Sequence, refs: Sequence) -> float:
    """Binary F1 — for two-class tasks (e.g. SST-2 sentiment)."""
    return float(f1_score(refs, preds, average="binary"))


@register("f1_macro")
def f1_macro(preds: Sequence, refs: Sequence) -> float:
    """Macro-averaged F1 — for multi-class / imbalanced tasks."""
    return float(f1_score(refs, preds, average="macro"))


def get(name: str) -> Callable[[Sequence, Sequence], float]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown metric '{name}'. Available: {available()}. "
            "Add it to eval_lib/metrics.py — do not write a one-off."
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def evaluate(preds: Sequence, refs: Sequence, metric_names: Sequence[str]) -> dict[str, float]:
    """Compute every named metric. Returns ``{metric_name: score}``."""
    return {name: get(name)(preds, refs) for name in metric_names}
