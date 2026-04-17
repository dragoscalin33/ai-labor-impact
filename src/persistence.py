"""
Lightweight reproducibility artifacts.

Saves a self-describing snapshot of every "official" run so anyone
(reviewer, CI, future-you) can reload the exact fit + seed + data version
that produced a given chart.

Two formats per save:
- ``<name>.pkl``  — full FitResult / numpy state for exact reload
- ``<name>.json`` — human-readable metadata for `git diff` / code review

Designed to be MLflow-independent so the project still has reproducibility
even on a minimal install.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ARTIFACTS_DIR, DATA_VERSION


@dataclass
class RunArtifact:
    """Container describing one reproducible run."""

    name: str
    created_at: str
    data_version: str
    seed: int
    n_samples: int
    benchmark: str
    fit_params: list[float]
    fit_param_names: list[str]
    fit_r_squared: float
    fit_rmse: float
    inflection_year: float
    year_to_99: float | None
    extra: dict[str, Any]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(
    name: str,
    fit_result: Any,
    seed: int,
    n_samples: int,
    benchmark: str = "swe_bench",
    extra: dict[str, Any] | None = None,
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> Path:
    """
    Persist a run's fit + metadata to ``artifacts_dir/<name>/``.

    Returns the directory path. Creates two files:
    - ``fit.pkl``     — the full ``FitResult`` for exact reload.
    - ``run.json``    — human-readable metadata.
    """
    out = _ensure_dir(Path(artifacts_dir) / name)

    # Pickle the FitResult — preserves numpy arrays, covariance, etc.
    with (out / "fit.pkl").open("wb") as f:
        pickle.dump(fit_result, f)

    y99 = fit_result.year_to_reach(0.99)
    artifact = RunArtifact(
        name=name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_version=DATA_VERSION,
        seed=seed,
        n_samples=n_samples,
        benchmark=benchmark,
        fit_params=fit_result.params.tolist(),
        fit_param_names=list(fit_result.param_names),
        fit_r_squared=float(fit_result.r_squared),
        fit_rmse=float(fit_result.rmse),
        inflection_year=float(fit_result.inflection_year()),
        year_to_99=float(y99) if y99 is not None else None,
        extra=extra or {},
    )

    # JSON for human readability + git diffs.
    payload = {
        **artifact.__dict__,
        "covariance": fit_result.covariance.tolist(),
        "ci_95": fit_result.ci_95.tolist(),
    }
    (out / "run.json").write_text(_safe_json_dumps(payload))
    return out


def load_run(name: str, artifacts_dir: str | Path = ARTIFACTS_DIR) -> tuple[Any, dict[str, Any]]:
    """
    Reload a saved run.

    Returns ``(fit_result, metadata_dict)``.
    """
    base = Path(artifacts_dir) / name
    with (base / "fit.pkl").open("rb") as f:
        fit_result = pickle.load(f)  # noqa: S301 — trusted local artifact
    metadata = json.loads((base / "run.json").read_text())
    return fit_result, metadata


def _safe_json_dumps(obj: Any) -> str:
    def default(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return str(x)

    return json.dumps(obj, indent=2, default=default)
