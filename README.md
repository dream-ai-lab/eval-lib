# eval-lib

The shared evaluation standard for the research group, packaged as an
installable library. Experiment repos depend on a **pinned version** of this
package — they never fork or copy it.

```bash
pip install "git+https://github.com/dream-ai-lab/eval-lib@v0.1.0"
```

## API

```python
from eval_lib import load_spec, run_paper, log_run, metrics

# In a classification experiment you only write model_fn(texts) -> list[int].
run_paper("eval_spec.yaml", model_fn, role="reproduce")
```

- `metrics` — the metric registry (`accuracy`, `f1`, `f1_macro`). Used by name
  from `eval_spec.yaml`. Add a *standard* metric here (with a test) and bump the
  version.
- `load_spec(path)` — load + validate an `eval_spec.yaml` (pinned dataset +
  model, known metrics, required fields).
- `log_run(spec, results, ...)` — the system of record: record a `results` dict
  (computed however you like) + provenance to MLflow. See below.
- `run_paper(spec, model_fn, role=..., extra_metrics=...)` — classification
  convenience over `log_run`: load the pinned dataset, run the model, compute
  metrics, then log.

## Experimental metrics — run a new metric now, no PR required

A paper may need a metric the registry doesn't ship yet. Instead of blocking the
run on a PR + release here, declare it in the spec and pass the callable at
runtime:

```yaml
# eval_spec.yaml
metrics:
  primary: "mcc"
  experimental: ["mcc"]     # not in the registry; supplied at runtime
```

```python
def mcc(preds, refs): ...   # your metric, (preds, refs) -> float

run_paper("eval_spec.yaml", model_fn, extra_metrics={"mcc": mcc})
```

The run is tagged `eval_tier=experimental` and records which metrics were
experimental plus a fingerprint of their implementation, so it is never mistaken
for a standard baseline. An experimental metric may **not** reuse a standard
metric's name. **Promote** it when it stabilises: add the function here, bump the
version, and drop it from `metrics.experimental` — the run then logs
`eval_tier=standard` automatically.

## System of record — log any result, not just classification

`run_paper` owns the classification eval loop (load split → model_fn → metrics).
But some evaluations don't fit that shape — generative win-rate, MT-Bench, an LLM
judge. For those, compute the numbers yourself and just record them:

```python
from eval_lib import log_run

results = {"alpaca_win_rate": 0.62, "mt_bench": 7.8}   # produced by YOUR harness
log_run(
    "eval_spec.yaml", results, role="proposal", parent_run_id=baseline,
    code_fingerprint=git_sha("eval/"),                 # identity of your eval code
    params={"judge": "gpt-4o-2024-08-06", "judge_temp": 0.0},
    artifacts=["eval/judge_prompt.txt", "outputs/generations.jsonl"],
)
```

`log_run` does not compute anything — it validates the spec (so dataset/model
provenance is still pinned) and writes the standard golden record. A run that
passes `code_fingerprint` (bring-your-own code) is tagged `eval_tier=experimental`;
`run_paper` passes `metric_lib_version` and is `eval_tier=standard`.

## Why this is a package, not a copy

A copied metric library drifts; pinned versions don't. Comparability is **logged,
not assumed**: two runs are comparable iff they share the same `eval_spec_hash`
**and** either the same `metric_lib_version` (blessed path) or the same
`code_fingerprint` (bring-your-own). Bumping a metric is a deliberate, reviewed
release here — that is what keeps standard numbers comparable across the org.

## Versioning

`eval_lib/version.py` is the single source of truth and must match the
`version` in `pyproject.toml`. Tag releases as `vX.Y.Z`.

## Develop

```bash
pip install -e ".[dev]" pytest    # or: pip install -e . pytest
PYTHONPATH="$PWD" pytest tests/ -q
```
