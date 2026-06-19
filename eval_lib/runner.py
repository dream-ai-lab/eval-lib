"""The eval runner + the system of record.

``log_run`` is the core: given a ``results`` dict you computed however you like,
it records the standard golden record (provenance, pinned spec, tier) to MLflow.
It does NOT compute anything — so any evaluation paradigm (classification,
generative win-rate, an LLM judge) can use the same store.

``run_paper`` is a convenience over ``log_run`` for the classification case: it
loads the pinned dataset, runs the caller's ``model_fn``, computes metrics from
the shared library, then delegates the logging to ``log_run``. Reproduce and
proposal runs use the SAME path — the only difference is the ``role`` tag and an
optional parent run id — which is what makes deltas comparable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Sequence

import mlflow

from . import metrics, version
from .data import load_eval_split
from .spec import (
    check_experimental_provided,
    experimental_names,
    load_spec,
    metric_names,
)

# model_fn takes the list of input texts and returns predicted labels.
ModelFn = Callable[[Sequence[str]], Sequence[int]]


def _git(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return ""


def _source_provenance() -> dict:
    """Where the code came from — so a run traces back to the exact source.

    commit hash alone is not enough: you also need the repo URL (which repo?)
    and whether the tree was dirty (does the hash represent what ran?). Each
    field accepts an env override (GIT_COMMIT / GIT_REMOTE / GIT_BRANCH /
    GIT_DIRTY) for containers / CI where ``.git`` is absent.
    """
    in_repo = _git(["rev-parse", "--is-inside-work-tree"]) == "true"
    dirty = os.environ.get("GIT_DIRTY")
    if dirty is None:
        dirty = ("true" if _git(["status", "--porcelain"]) else "false") if in_repo else "unknown"
    return {
        "git_commit": os.environ.get("GIT_COMMIT") or _git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "git_remote": os.environ.get("GIT_REMOTE") or _git(["config", "--get", "remote.origin.url"]) or "unknown",
        "git_branch": os.environ.get("GIT_BRANCH") or _git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "git_dirty": dirty,
    }


def check_target(results: dict, target) -> bool:
    """True iff every metric in ``reproduce_target`` lands within [min, max]."""
    target = target or {}
    for name, rng in vars(target).items() if hasattr(target, "__dict__") else target.items():
        rng = vars(rng) if hasattr(rng, "__dict__") else rng
        v = results.get(name)
        if v is None or not (rng["min"] <= v <= rng["max"]):
            return False
    return True


def _config_params(spec) -> dict:
    """Every config scalar, flattened, so each is individually searchable in
    MLflow (a teammate can filter on dataset.revision, inference.seed, etc.)."""
    ds, m, inf = spec.dataset, spec.model, spec.inference
    return {
        "spec_version": getattr(spec, "spec_version", "unknown"),
        "task": getattr(spec, "task", ""),
        "dataset.hf_id": ds.hf_id,
        "dataset.config": getattr(ds, "config", None),
        "dataset.split": ds.split,
        "dataset.version": ds.version,
        "dataset.revision": getattr(ds, "revision", None),
        "dataset.text_field": ds.text_field,
        "dataset.label_field": ds.label_field,
        "model.hf_id": m.hf_id,
        "model.revision": m.revision,
        "inference.seed": inf.seed,
        "inference.max_length": inf.max_length,
        "inference.batch_size": inf.batch_size,
        "metrics.primary": spec.metrics.primary,
        "metrics.secondary": ",".join(getattr(spec.metrics, "secondary", []) or []),
    }


def _snapshot_entry_script() -> None:
    """Attach the entry script (``sys.argv[0]``) as an artifact, so the run stays
    traceable even if its source repo is later moved, deleted, or force-pushed."""
    entry = sys.argv[0] if sys.argv else ""
    if entry and os.path.isfile(entry):
        try:
            mlflow.log_artifact(entry, artifact_path="source")
        except Exception:
            pass


def _log_deltas(results: dict, parent_run_id: str) -> None:
    """Log ``delta_<metric>`` against the baseline (parent) run, for proposals."""
    base = mlflow.get_run(parent_run_id).data.metrics
    for k, v in results.items():
        if k in base:
            mlflow.log_metric(f"delta_{k}", v - base[k])


def log_run(
    spec_path: str,
    results: dict[str, float],
    *,
    role: str = "reproduce",
    parent_run_id: str | None = None,
    run_name: str | None = None,
    metric_lib_version: str | None = None,
    code_fingerprint: str | None = None,
    params: dict | None = None,
    artifacts: list[str] | None = None,
    extra_tags: dict | None = None,
) -> dict:
    """Record a result + provenance to MLflow — the system of record.

    eval-lib does NOT compute ``results`` here: you produce them however you like
    (classification via ``run_paper``, generative win-rate, an LLM judge, …) and
    this logs the standard golden record so the run is traceable and its
    comparability is *legible*:

    - ``eval_tier=standard`` — produced via the blessed path (no bring-your-own
      code); pass ``metric_lib_version``.
    - ``eval_tier=experimental`` — bring-your-own code; pass ``code_fingerprint``.
      Never a standard baseline.

    Comparability (logged, not enforced): two runs compare iff same
    ``eval_spec_hash`` AND (same ``metric_lib_version`` OR same
    ``code_fingerprint``). The spec is still validated (pinned dataset/model,
    hash) — provenance is mandatory even when the compute path is your own.

    ``params``/``artifacts``/``extra_tags`` attach extra searchable config (e.g.
    judge model, decoding), files (e.g. generations), and tags.

    Returns ``{run_id, results, reproduce_passed}``.
    """
    spec = load_spec(spec_path)
    passed = check_target(results, spec.reproduce_target)
    tier = "experimental" if code_fingerprint else "standard"

    golden = {
        "paper_id": spec.paper_id,
        **_source_provenance(),
        "hf_dataset_id": f"{spec.dataset.hf_id}@{spec.dataset.version}",
        "eval_spec_hash": spec.hash,
        "metric_lib_version": metric_lib_version or "byo",
    }

    mlflow.set_experiment(spec.paper_id)
    with mlflow.start_run(run_name=run_name or f"{role}-{spec.paper_id}") as active:
        mlflow.log_params(golden)
        mlflow.log_param("hf_model_id", f"{spec.model.hf_id}@{spec.model.revision}")
        mlflow.log_params(params or {})
        # The exact spec file, attached so anyone can pull the full config.
        mlflow.log_artifact(spec_path, artifact_path="eval_spec")
        _snapshot_entry_script()
        for path in artifacts or []:
            if os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path="extra")
        tags = {
            **golden,
            "role": role,
            "reproduce_passed": str(passed),
            "eval_tier": tier,
        }
        # The fingerprint pins the code that produced the numbers, so two
        # experimental runs are comparable only when the code — not just the
        # metric name — matches.
        if code_fingerprint:
            tags["code_fingerprint"] = code_fingerprint
        tags.update(extra_tags or {})
        mlflow.set_tags(tags)
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)
        mlflow.log_metrics(results)

        if role == "proposal" and parent_run_id:
            _log_deltas(results, parent_run_id)

        run_id = active.info.run_id

    return {"run_id": run_id, "results": results, "reproduce_passed": passed}


def run_paper(
    spec_path: str,
    model_fn: ModelFn,
    role: str = "reproduce",
    parent_run_id: str | None = None,
    run_name: str | None = None,
    extra_metrics: dict[str, Callable] | None = None,
) -> dict:
    """Classification convenience over ``log_run``: load the pinned split → run
    ``model_fn`` → compute blessed metrics → log.

    ``extra_metrics`` supplies the callables for any metric declared under
    ``metrics.experimental`` in the spec (``{name: fn}``). Such a run is tagged
    ``eval_tier=experimental`` and is never a standard baseline.

    Returns ``{run_id, results, reproduce_passed}``.
    """
    spec = load_spec(spec_path)
    names = metric_names(spec)

    # Experimental metrics live in the experiment, not the registry. Cross-check
    # the declaration against what was actually passed before doing any work.
    experimental = experimental_names(spec)
    extra_metrics = extra_metrics or {}
    check_experimental_provided(experimental, extra_metrics)

    texts, refs = load_eval_split(spec)
    preds = list(model_fn(texts))
    if len(preds) != len(refs):
        raise ValueError(f"model_fn returned {len(preds)} preds for {len(refs)} examples")

    results = metrics.evaluate(preds, refs, names, extra=extra_metrics)

    return log_run(
        spec_path,
        results,
        role=role,
        parent_run_id=parent_run_id,
        run_name=run_name,
        metric_lib_version=version.__version__,
        code_fingerprint=metrics.fingerprint(extra_metrics) if experimental else None,
        params=_config_params(spec),
        extra_tags=(
            {"experimental_metrics": ",".join(sorted(experimental))} if experimental else None
        ),
    )
