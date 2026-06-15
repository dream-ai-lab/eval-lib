"""The standard eval runner.

Every experiment goes through ``run_paper``. It loads the pinned dataset,
runs the caller's ``model_fn``, computes metrics from the shared library,
and logs the golden record to MLflow. Reproduce and proposal runs use the
SAME path — the only difference is the ``role`` tag and an optional parent
run id — which is what makes deltas comparable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Sequence

import mlflow

from . import metrics, version
from .data import load_eval_split
from .spec import load_spec, metric_names

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


def run_paper(
    spec_path: str,
    model_fn: ModelFn,
    role: str = "reproduce",
    parent_run_id: str | None = None,
    run_name: str | None = None,
) -> dict:
    """Validate spec → load data → run model → log golden record to MLflow.

    Returns ``{run_id, results, reproduce_passed}``.
    """
    spec = load_spec(spec_path)
    names = metric_names(spec)

    texts, refs = load_eval_split(spec)
    preds = list(model_fn(texts))
    if len(preds) != len(refs):
        raise ValueError(f"model_fn returned {len(preds)} preds for {len(refs)} examples")

    results = metrics.evaluate(preds, refs, names)
    passed = check_target(results, spec.reproduce_target)

    # The five required golden-record fields (logged as params AND tags so
    # they show up both in run detail and in filterable search).
    golden = {
        "paper_id": spec.paper_id,
        **_source_provenance(),
        "hf_dataset_id": f"{spec.dataset.hf_id}@{spec.dataset.version}",
        "eval_spec_hash": spec.hash,
        "metric_lib_version": version.__version__,
    }

    mlflow.set_experiment(spec.paper_id)
    with mlflow.start_run(run_name=run_name or f"{role}-{spec.paper_id}") as active:
        mlflow.log_params(golden)
        mlflow.log_param("hf_model_id", f"{spec.model.hf_id}@{spec.model.revision}")
        mlflow.log_params(_config_params(spec))
        # The exact spec file, attached so anyone can pull the full config.
        mlflow.log_artifact(spec_path, artifact_path="eval_spec")
        # Snapshot the entry script too, so the run stays traceable even if the
        # source repo is later moved, deleted, or force-pushed.
        entry = sys.argv[0] if sys.argv else ""
        if entry and os.path.isfile(entry):
            try:
                mlflow.log_artifact(entry, artifact_path="source")
            except Exception:
                pass
        mlflow.set_tags(
            {
                **golden,
                "role": role,
                "reproduce_passed": str(passed),
            }
        )
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)
        mlflow.log_metrics(results)

        # For a proposal, log the delta vs the baseline (parent) run.
        if role == "proposal" and parent_run_id:
            base = mlflow.get_run(parent_run_id).data.metrics
            for k, v in results.items():
                if k in base:
                    mlflow.log_metric(f"delta_{k}", v - base[k])

        run_id = active.info.run_id

    return {"run_id": run_id, "results": results, "reproduce_passed": passed}
