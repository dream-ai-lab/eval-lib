"""Load, validate, and hash an ``eval_spec.yaml``.

The spec is the *contract* for a paper: which dataset (pinned), which
model (pinned), which metrics, and what counts as a successful
reproduce. Validation here is what turns the convention into a standard —
a malformed spec, or one referencing an unknown metric, is rejected
before any run happens (locally and in CI).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import yaml

from . import metrics

# Fields that MUST be logged on every run (golden record).
REQUIRED_LOG_FIELDS = [
    "paper_id",
    "git_commit",
    "hf_dataset_id",
    "eval_spec_hash",
    "metric_lib_version",
]


class SpecError(ValueError):
    """Raised when an eval_spec.yaml violates the standard."""


def canonical_hash(spec_dict: dict) -> str:
    """Hash the spec independent of key order / whitespace.

    Reformatting the YAML (reordering keys, changing indentation) must NOT
    change the hash — only a semantic change should. We achieve that by
    serialising with sorted keys and no insignificant whitespace.
    """
    canon = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return "sha256:" + digest[:16]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def validate(raw: dict) -> None:
    """Enforce the standard. Raises SpecError on the first violation."""
    _require("paper_id" in raw, "missing 'paper_id'")

    ds = raw.get("dataset", {})
    _require(bool(ds.get("hf_id")), "dataset.hf_id is required")
    _require(bool(ds.get("split")), "dataset.split is required")
    _require(
        ds.get("version") not in (None, "", "latest"),
        "dataset.version must be pinned (not 'latest' or empty)",
    )
    _require(bool(ds.get("text_field")), "dataset.text_field is required")
    _require(bool(ds.get("label_field")), "dataset.label_field is required")

    model = raw.get("model", {})
    _require(bool(model.get("hf_id")), "model.hf_id is required")
    _require(
        model.get("revision") not in (None, "", "main"),
        "model.revision must be pinned to a commit/tag (not 'main' or empty)",
    )

    m = raw.get("metrics", {})
    primary = m.get("primary")
    _require(bool(primary), "metrics.primary is required")
    available = metrics.available()

    # Experimental metrics are supplied at runtime (run_paper(extra_metrics=...))
    # instead of living in the registry, so a new-paper metric never blocks a run
    # on a PR here. They may not shadow a standard metric, and any run using one
    # is tagged eval_tier=experimental.
    experimental = m.get("experimental", []) or []
    _require(isinstance(experimental, list), "metrics.experimental must be a list")
    for name in experimental:
        _require(
            name not in available,
            f"metrics.experimental '{name}' shadows a standard metric; "
            "remove it from 'experimental' (use the registered one)",
        )
    allowed = set(available) | set(experimental)

    _require(
        primary in allowed,
        f"metrics.primary '{primary}' not in metric_lib {available} "
        "nor declared in metrics.experimental",
    )
    for sec in m.get("secondary", []) or []:
        _require(
            sec in allowed,
            f"metrics.secondary '{sec}' not in metric_lib {available} "
            "nor declared in metrics.experimental",
        )

    _require("reproduce_target" in raw, "missing 'reproduce_target'")
    for name, rng in raw["reproduce_target"].items():
        _require(
            isinstance(rng, dict) and "min" in rng and "max" in rng,
            f"reproduce_target.{name} must be {{min, max}}",
        )


def _ns(obj):
    """Recursively wrap dicts so callers can use attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def load_spec(path: str):
    """Load + validate a spec. Returns a namespace with a ``.hash`` and
    ``.raw`` (the original dict) attached."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    validate(raw)
    spec = _ns(raw)
    spec.hash = canonical_hash(raw)
    spec.raw = raw
    return spec


def metric_names(spec) -> list[str]:
    names = [spec.metrics.primary]
    names += list(getattr(spec.metrics, "secondary", []) or [])
    return names


def experimental_names(spec) -> list[str]:
    """Metrics declared experimental in the spec (supplied at runtime)."""
    return list(getattr(spec.metrics, "experimental", []) or [])


def check_experimental_provided(experimental, provided) -> None:
    """Cross-check declared experimental metrics against runtime callables.

    Every name in ``metrics.experimental`` must have a callable passed to
    ``run_paper(extra_metrics=...)``; a callable passed but not declared is an
    error (a typo, or an attempt to override a standard metric). Raises
    ``SpecError`` on the first mismatch.
    """
    experimental, provided = set(experimental or []), set(provided or [])
    missing = experimental - provided
    _require(
        not missing,
        f"metrics.experimental declares {sorted(missing)} but no callable was "
        "passed to run_paper(extra_metrics=...)",
    )
    stray = provided - experimental
    _require(
        not stray,
        f"extra_metrics passed {sorted(stray)} but they are not declared in "
        "metrics.experimental (add them, or remove the override)",
    )
