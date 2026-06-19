# eval-lib

The shared evaluation standard for the research group, packaged as an
installable library. Experiment repos depend on a **pinned version** of this
package — they never fork or copy it.

```bash
pip install "git+https://github.com/dream-ai-lab/eval-lib@v0.1.0"
```

## API

```python
from eval_lib import load_spec, run_paper, metrics

# In an experiment repo you only write model_fn(texts) -> list[int].
run_paper("eval_spec.yaml", model_fn, role="reproduce")
```

- `metrics` — the metric registry (`accuracy`, `f1`, `f1_macro`). Used by name
  from `eval_spec.yaml`. Add a *standard* metric here (with a test) and bump the
  version.
- `load_spec(path)` — load + validate an `eval_spec.yaml` (pinned dataset +
  model, known metrics, required fields).
- `run_paper(spec, model_fn, role=..., extra_metrics=...)` — load the pinned
  dataset, run the model, log the golden record to MLflow, check the reproduce
  target.

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

## Why this is a package, not a copy

A copied metric library drifts; pinned versions don't. Because every run logs
`metric_lib_version`, two runs are only comparable when they used the same
version of this package. Bumping a metric is a deliberate, reviewed release
here — that is what keeps numbers comparable across the whole org.

## Versioning

`eval_lib/version.py` is the single source of truth and must match the
`version` in `pyproject.toml`. Tag releases as `vX.Y.Z`.

## Develop

```bash
pip install -e ".[dev]" pytest    # or: pip install -e . pytest
PYTHONPATH="$PWD" pytest tests/ -q
```
