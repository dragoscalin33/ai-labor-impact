"""
Experiment tracking utilities.

Wraps MLflow with project-aware helpers so each model fit and Monte Carlo
run produces a reproducible, auditable artifact in ``mlruns/``.

MLflow itself is a soft dependency — imported lazily so a plain dashboard
deploy doesn't pay the install cost.
"""

from .mlflow_tracker import (
    MLflowTracker,
    log_benchmark_fit,
    log_scenario_run,
    track_run,
)

__all__ = [
    "MLflowTracker",
    "log_benchmark_fit",
    "log_scenario_run",
    "track_run",
]
