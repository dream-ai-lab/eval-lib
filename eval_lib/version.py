"""Single source of truth for the eval_lib version.

This version is logged with EVERY run. If anyone changes a metric
implementation, bump this — past scores computed with a different
version are not directly comparable.

0.1.1 — traceability only (git_remote/branch/dirty + entry-script artifact).
        Metrics unchanged → scores remain comparable with 0.1.0.
"""

__version__ = "0.1.1"
