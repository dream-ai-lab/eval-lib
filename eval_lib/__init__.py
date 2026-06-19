"""eval_lib — the shared evaluation standard for the research group.

Public API:
    load_spec(path)              load + validate an eval_spec.yaml
    log_run(spec, results)       record a result + provenance to MLflow
    run_paper(spec, model_fn)    classification convenience over log_run
    metrics.evaluate(...)        compute metrics by name
"""

from . import metrics
from .runner import log_run, run_paper
from .spec import SpecError, load_spec, metric_names
from .version import __version__

__all__ = [
    "load_spec",
    "log_run",
    "run_paper",
    "metrics",
    "metric_names",
    "SpecError",
    "__version__",
]
