"""Shared metric library — the home of every *standard* metric.

Every experiment imports metrics from here by NAME (referenced in
``eval_spec.yaml``). This is what makes numbers comparable across people and
across papers.

To add a standard metric: write a function ``(preds, refs) -> float`` and
decorate it with ``@register("name")``. Then it is usable from any eval_spec by
that name, and CI will accept specs that reference it.

A paper may need a metric we do not ship yet. Rather than block the run on a PR
here, an experiment may declare it under ``metrics.experimental`` and pass the
callable to ``run_paper(extra_metrics=...)``; that run is tagged
``eval_tier=experimental`` so it cannot be mistaken for a standard baseline.
Promote it by adding it here and bumping the version.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Callable, Mapping, Sequence

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


def evaluate(
    preds: Sequence,
    refs: Sequence,
    metric_names: Sequence[str],
    extra: Mapping[str, Callable[[Sequence, Sequence], float]] | None = None,
) -> dict[str, float]:
    """Compute every named metric. Returns ``{metric_name: score}``.

    ``extra`` supplies experimental metrics by name — callables that are not in
    the registry. For a name present in ``extra`` the callable is used in
    preference to the registry; any other name resolves through ``get``.
    """
    extra = extra or {}
    return {
        name: (extra[name] if name in extra else get(name))(preds, refs)
        for name in metric_names
    }


def fingerprint(extra_metrics: Mapping[str, Callable]) -> str:
    """Stable hash of experimental metric *implementations*.

    Two experimental runs that name the same metric are only comparable when the
    implementation matches. Hashing each callable's source makes that checkable;
    when source is unavailable (e.g. a C builtin) we fall back to its qualified
    name. Returns a ``sha256:<16 hex>`` digest, stable across key order.
    """
    parts = []
    for name in sorted(extra_metrics):
        fn = extra_metrics[name]
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            src = getattr(fn, "__qualname__", repr(fn))
        parts.append(f"{name}:{src}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return "sha256:" + digest[:16]
