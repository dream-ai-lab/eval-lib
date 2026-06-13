"""eval_lib — the shared evaluation standard for the research group.

Public API:
    load_spec(path)              load + validate an eval_spec.yaml
    run_paper(spec, model_fn)    run an experiment and log to MLflow
    metrics.evaluate(...)        compute metrics by name
"""

from . import metrics
from .runner import run_paper
from .spec import SpecError, load_spec, metric_names
from .version import __version__

__all__ = [
    "load_spec",
    "run_paper",
    "metrics",
    "metric_names",
    "SpecError",
    "__version__",
]
