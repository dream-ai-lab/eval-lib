"""Single source of truth for the eval_lib version.

This version is logged with EVERY run. If anyone changes a metric
implementation, bump this — past scores computed with a different
version are not directly comparable.

0.1.1 — traceability only (git_remote/branch/dirty + entry-script artifact).
        Metrics unchanged → scores remain comparable with 0.1.0.
0.2.0 — experimental metrics: a spec may declare metrics.experimental and pass
        the callables to run_paper(extra_metrics=...); such runs are tagged
        eval_tier=experimental. Standard metrics unchanged → standard-tier
        scores remain comparable with 0.1.x.
0.3.0 — system of record: log_run(spec, results) records any result + provenance
        to MLflow regardless of how it was computed (run_paper now delegates to
        it). Comparability is logged via metric_lib_version OR code_fingerprint.
        Tag rename: experimental_metrics_fingerprint -> code_fingerprint.
"""

__version__ = "0.3.0"
