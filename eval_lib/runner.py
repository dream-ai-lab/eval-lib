"""The eval runner + the system of record.

``log_run`` is the core: given a ``results`` dict you computed however you like,
it records the standard golden record (provenance, pinned spec, tier) to Weights
& Biases. It does NOT compute anything — so any evaluation paradigm
(classification, generative win-rate, an LLM judge) can use the same store.

``run_paper`` is a convenience over ``log_run`` for the classification case: it
loads the pinned dataset, runs the caller's ``model_fn``, computes metrics from
the shared library, then delegates the logging to ``log_run``. Reproduce and
proposal runs use the SAME path — the only difference is the ``role`` job type
and an optional parent run id — which is what makes deltas comparable.

W&B target is taken from the environment: ``WANDB_ENTITY`` (team) and
``WANDB_PROJECT`` (defaults to ``eval-lib``); every run is grouped by
``paper_id`` so a paper's runs sit together. ``wandb login`` / ``WANDB_API_KEY``
controls auth; ``WANDB_MODE=offline`` runs without a network.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Sequence

import wandb

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

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "eval-lib")


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
    W&B (a teammate can filter on dataset.revision, inference.seed, etc.)."""
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


def _attach_files(run, paths) -> None:
    """Attach files (spec, entry script, extras) to the run, so it stays
    traceable even if its source repo is later moved, deleted, or force-pushed."""
    for path in paths:
        if path and os.path.isfile(path):
            try:
                run.save(path, base_path=os.path.dirname(os.path.abspath(path)), policy="now")
            except Exception:
                pass


def _log_deltas(run, results: dict, parent_run_id: str) -> None:
    """Best-effort ``delta_<metric>`` vs the baseline (parent) run.

    Needs the W&B API (online); silently skipped offline or without access."""
    try:
        parent = wandb.Api().run(f"{run.entity}/{run.project}/{parent_run_id}")
        base = dict(parent.summary)
        for k, v in results.items():
            if k in base:
                run.summary[f"delta_{k}"] = v - base[k]
    except Exception:
        pass


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
    """Record a result + provenance to W&B — the system of record.

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

    # The golden record — provenance + pinned identity — lives in W&B config so
    # every field is individually filterable.
    config = {
        "paper_id": spec.paper_id,
        **_source_provenance(),
        "hf_dataset_id": f"{spec.dataset.hf_id}@{spec.dataset.version}",
        "hf_model_id": f"{spec.model.hf_id}@{spec.model.revision}",
        "eval_spec_hash": spec.hash,
        "metric_lib_version": metric_lib_version or "byo",
        "role": role,
        "eval_tier": tier,
        "reproduce_passed": passed,
        **(params or {}),
        **(extra_tags or {}),
    }
    if code_fingerprint:
        config["code_fingerprint"] = code_fingerprint
    if parent_run_id:
        config["parent_run_id"] = parent_run_id

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=WANDB_PROJECT,
        group=spec.paper_id,          # a paper's runs sit together (≈ an experiment)
        job_type=role,
        name=run_name or f"{role}-{spec.paper_id}",
        config=config,
        tags=[f"role:{role}", f"tier:{tier}", spec.paper_id],
        reinit=True,
    )
    try:
        run.log(results)              # final metrics → run summary + history point
        _attach_files(run, [spec_path, sys.argv[0] if sys.argv else "", *(artifacts or [])])
        if role == "proposal" and parent_run_id:
            _log_deltas(run, results, parent_run_id)
        run_id = run.id
    finally:
        run.finish()

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
