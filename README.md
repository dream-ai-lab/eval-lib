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
  from `eval_spec.yaml`. Add a metric here (with a test) and bump the version —
  never write a one-off eval function in an experiment repo.
- `load_spec(path)` — load + validate an `eval_spec.yaml` (pinned dataset +
  model, known metrics, required fields).
- `run_paper(spec, model_fn, role=...)` — load the pinned dataset, run the
  model, log the golden record to MLflow, check the reproduce target.

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
