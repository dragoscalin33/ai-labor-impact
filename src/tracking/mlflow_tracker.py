"""
MLflow experiment tracking for the AI-labor-impact pipeline.

Why
---
Every figure in the dashboard is the output of (a) a sigmoid fit to a
specific benchmark snapshot and (b) a Monte Carlo simulation with a
specific seed. Without tracking, "the chart from yesterday" is a black
box — you cannot tell *which* fit produced it or whether re-running with
the same seed would reproduce it.

This module logs every fit + scenario run as an MLflow run with:
- params : data version, seed, n_samples, sigmoid bounds, scenario keys
- metrics: R², RMSE, inflection year, year_to_99, peak unemployment per scenario
- artifacts: serialized FitResult (pickle + JSON), tidy result DataFrames

MLflow is imported lazily — the dashboard does *not* require it at runtime.
Run ``mlflow ui`` from the project root to inspect history.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    DATA_VERSION,
    DEFAULT_N_SAMPLES,
    DEFAULT_RANDOM_SEED,
)

if TYPE_CHECKING:
    from src.models.benchmark_curve import FitResult

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "ai-labor-impact"


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is an optional dependency. Install with "
            "`pip install mlflow` (or use the `requirements-dev.txt`) "
            "to enable experiment tracking."
        ) from exc
    return mlflow


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class MLflowTracker:
    """
    Project-aware MLflow wrapper.

    Example
    -------
    >>> tracker = MLflowTracker(experiment="ai-labor-impact")
    >>> with tracker.start_run("swe_bench_sigmoid_fit") as run:
    ...     fitter = BenchmarkCurveFitter(model="sigmoid")
    ...     fit = fitter.fit(years, scores)
    ...     log_benchmark_fit(fit, benchmark="swe_bench")
    """

    def __init__(
        self,
        experiment: str = DEFAULT_EXPERIMENT_NAME,
        tracking_uri: str | None = None,
    ) -> None:
        mlflow = _import_mlflow()
        self.mlflow = mlflow
        self.tracking_uri = (
            tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI")
            or f"file:{Path.cwd() / 'mlruns'}"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment)
        self.experiment = experiment

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        base_tags = {"data_version": DATA_VERSION, "project": "ai-labor-impact"}
        if tags:
            base_tags.update(tags)
        with self.mlflow.start_run(run_name=run_name, tags=base_tags) as run:
            yield run


# Module-level convenience: a default tracker activated on first call.
_default_tracker: MLflowTracker | None = None


def _get_default_tracker() -> MLflowTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = MLflowTracker()
    return _default_tracker


@contextmanager
def track_run(run_name: str, **tags: str) -> Iterator[Any]:
    """Convenience wrapper around the default tracker."""
    with _get_default_tracker().start_run(run_name, tags=tags) as run:
        yield run


# ---------------------------------------------------------------------------
# Logging primitives
# ---------------------------------------------------------------------------

def log_benchmark_fit(
    fit: FitResult,
    benchmark: str,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """
    Log a fitted benchmark curve to the active MLflow run.

    Logged params: model_name, benchmark, n_points, data_version,
                   sigmoid_L, sigmoid_k, sigmoid_x0 (or gompertz equivalents).
    Logged metrics: r_squared, rmse, inflection_year, year_to_99.
    Logged artifacts: fit_result.pkl, fit_result.json, fit_summary.csv.
    """
    mlflow = _import_mlflow()

    mlflow.log_param("benchmark", benchmark)
    mlflow.log_param("model_name", fit.model_name)
    mlflow.log_param("n_points", fit.n_points)
    mlflow.log_param("data_version", DATA_VERSION)
    if extra_params:
        for k, v in extra_params.items():
            mlflow.log_param(k, v)
    for name, value in zip(fit.param_names, fit.params):
        safe = name.split()[0].lower()  # "L (asymptote)" -> "l"
        mlflow.log_param(f"param_{safe}", float(value))

    mlflow.log_metric("r_squared", float(fit.r_squared))
    mlflow.log_metric("rmse", float(fit.rmse))
    mlflow.log_metric("inflection_year", float(fit.inflection_year()))
    y99 = fit.year_to_reach(0.99)
    if y99 is not None:
        mlflow.log_metric("year_to_99pct", float(y99))

    artifacts_dir = Path(ARTIFACTS_DIR) / "tmp"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = artifacts_dir / f"fit_{benchmark}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(fit, f)
    mlflow.log_artifact(str(pkl_path))

    json_path = artifacts_dir / f"fit_{benchmark}.json"
    payload = {
        "model_name": fit.model_name,
        "param_names": list(fit.param_names),
        "params": fit.params.tolist(),
        "covariance": fit.covariance.tolist(),
        "ci_95": fit.ci_95.tolist(),
        "r_squared": float(fit.r_squared),
        "rmse": float(fit.rmse),
        "n_points": int(fit.n_points),
        "data_version": DATA_VERSION,
    }
    json_path.write_text(json.dumps(_to_json_safe(payload), indent=2))
    mlflow.log_artifact(str(json_path))

    csv_path = artifacts_dir / f"fit_{benchmark}_summary.csv"
    fit.summary().to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path))


def log_scenario_run(
    summary_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Log a Monte Carlo scenario run to the active MLflow run.

    Logged params: n_samples, seed, scenario_keys.
    Logged metrics: peak_unemployment_pct__<scenario_key> for each.
    Logged artifacts: scenario_summary.csv, peak_unemployment.csv.
    """
    mlflow = _import_mlflow()

    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("seed", seed)
    mlflow.log_param("data_version", DATA_VERSION)
    mlflow.log_param(
        "scenario_keys",
        ",".join(sorted(summary_df["scenario_key"].unique())),
    )

    for _, row in peak_df.iterrows():
        key = (
            str(row["scenario_name"])
            .lower()
            .replace(" ", "_")
            .replace("—", "")
            .replace("__", "_")
        )
        mlflow.log_metric(f"peak_unemployment_pct__{key}", float(row["peak_unemployment_pct"]))

    artifacts_dir = Path(ARTIFACTS_DIR) / "tmp"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sum_path = artifacts_dir / "scenario_summary.csv"
    peak_path = artifacts_dir / "peak_unemployment.csv"
    summary_df.to_csv(sum_path, index=False)
    peak_df.to_csv(peak_path, index=False)
    mlflow.log_artifact(str(sum_path))
    mlflow.log_artifact(str(peak_path))


def log_dataclass_params(obj: Any, prefix: str = "") -> None:
    """Best-effort log of a dataclass instance as MLflow params."""
    if not is_dataclass(obj):
        return
    mlflow = _import_mlflow()
    for k, v in asdict(obj).items():
        if isinstance(v, (int, float, str, bool)):
            mlflow.log_param(f"{prefix}{k}", v)
