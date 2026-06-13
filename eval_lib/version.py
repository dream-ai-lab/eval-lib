"""Single source of truth for the eval_lib version.

This version is logged with EVERY run. If anyone changes a metric
implementation, bump this — past scores computed with a different
version are not directly comparable.
"""

__version__ = "0.1.0"
