"""
AI Benchmark Curve Fitter
==========================
Fits a sigmoid (logistic) curve to historical AI benchmark data using
non-linear least squares regression (scipy.optimize.curve_fit).

Key output:
- Fitted parameters with 95% confidence intervals
- Projected AI capability score for any future year
- Projection uncertainty bands

This is fundamentally more rigorous than manually setting parameters:
the curve is *calibrated* against real published benchmark scores,
including the Claude Mythos Preview (SWE-bench 93.9%, April 2026).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

from src.config import (
    GOMPERTZ_BOUNDS_LOWER,
    GOMPERTZ_BOUNDS_UPPER,
    SIGMOID_BOUNDS_LOWER,
    SIGMOID_BOUNDS_UPPER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sigmoid model
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """
    Standard logistic (sigmoid) function.

    Parameters
    ----------
    x  : input values (years)
    L  : upper asymptote (max capability, typically 1.0)
    k  : growth rate
    x0 : inflection point (year of fastest growth)
    """
    return L / (1.0 + np.exp(-k * (x - x0)))


def gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Gompertz curve — an alternative to the logistic for asymmetric growth.
    Grows faster early, slower near saturation. Often fits technology
    adoption better than a symmetric logistic.
    """
    return a * np.exp(-b * np.exp(-c * x))


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Container for curve fit results."""
    params: np.ndarray          # [L, k, x0]
    param_names: list[str]
    covariance: np.ndarray
    std_errors: np.ndarray
    ci_95: np.ndarray           # shape (3, 2) — [lower, upper] per param
    r_squared: float
    rmse: float
    n_points: int
    model_name: str

    def predict(self, years: np.ndarray) -> np.ndarray:
        """Point prediction for given years."""
        if self.model_name == "sigmoid":
            return sigmoid(years, *self.params)
        elif self.model_name == "gompertz":
            return gompertz(years, *self.params)
        raise ValueError(f"Unknown model: {self.model_name}")

    def predict_with_uncertainty(
        self, years: np.ndarray, n_samples: int = 2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo uncertainty propagation using parameter covariance.

        Returns (mean, lower_95, upper_95) arrays.
        """
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(self.params, self.covariance, size=n_samples)

        preds = np.array([
            sigmoid(years, *s) if self.model_name == "sigmoid"
            else gompertz(years, *s)
            for s in samples
        ])
        # Clip to valid [0, 1] range
        preds = np.clip(preds, 0, 1)

        mean = preds.mean(axis=0)
        lower = np.percentile(preds, 2.5, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        return mean, lower, upper

    def inflection_year(self) -> float:
        """Year of maximum growth rate (sigmoid x0 parameter)."""
        if self.model_name == "sigmoid":
            return float(self.params[2])  # x0
        # For Gompertz: inflection at x = ln(b)/c
        a, b, c = self.params
        return float(np.log(b) / c)

    def year_to_reach(self, target: float) -> float | None:
        """
        Estimate the year when capability crosses `target` (0–1 scale).
        Returns None if target is unreachable.
        """
        if target >= self.params[0]:
            return None
        if self.model_name == "sigmoid":
            L, k, x0 = self.params
            return float(x0 - np.log(L / target - 1) / k)
        return None

    def summary(self) -> pd.DataFrame:
        """Human-readable parameter summary with confidence intervals."""
        rows = []
        for i, name in enumerate(self.param_names):
            rows.append({
                "parameter": name,
                "estimate": self.params[i],
                "std_error": self.std_errors[i],
                "ci_95_lower": self.ci_95[i, 0],
                "ci_95_upper": self.ci_95[i, 1],
            })
        return pd.DataFrame(rows)


class BenchmarkCurveFitter:
    """
    Fits sigmoid or Gompertz curves to AI benchmark data.

    Example
    -------
    >>> from src.data.ai_benchmarks import get_swe_bench_series
    >>> years, scores = get_swe_bench_series()
    >>> fitter = BenchmarkCurveFitter()
    >>> result = fitter.fit(years, scores)
    >>> print(result.summary())
    >>> print(f"Inflection year: {result.inflection_year():.1f}")
    >>> print(f"Year to reach 99%: {result.year_to_reach(0.99):.1f}")
    """

    MODELS = {"sigmoid", "gompertz"}

    def __init__(self, model: str = "sigmoid") -> None:
        if model not in self.MODELS:
            raise ValueError(f"model must be one of {self.MODELS}")
        self.model = model

    def fit(
        self,
        years: np.ndarray,
        scores: np.ndarray,
        p0: list[float] | None = None,
    ) -> FitResult:
        """
        Fit the capability curve to (years, scores) data.

        Parameters
        ----------
        years  : array of decimal years (e.g. [2021.5, 2023.25, ...])
        scores : array of normalized scores in [0, 1]
        p0     : initial parameter guess [L, k, x0] for sigmoid

        Returns
        -------
        FitResult dataclass with params, CIs, and prediction methods.
        """
        years = np.asarray(years, dtype=float)
        scores = np.asarray(scores, dtype=float)

        func: Callable[..., np.ndarray]
        if self.model == "sigmoid":
            func = sigmoid
            param_names = ["L (asymptote)", "k (growth rate)", "x0 (inflection year)"]
            if p0 is None:
                # Smart initial guess: asymptote slightly above max observed,
                # inflection near the data midpoint
                p0 = [min(scores.max() * 1.05, 1.0), 0.5, float(np.median(years))]
            bounds = (list(SIGMOID_BOUNDS_LOWER), list(SIGMOID_BOUNDS_UPPER))
        else:
            func = gompertz
            param_names = ["a (asymptote)", "b (displacement)", "c (growth rate)"]
            if p0 is None:
                p0 = [1.0, 10.0, 0.5]
            bounds = (list(GOMPERTZ_BOUNDS_LOWER), list(GOMPERTZ_BOUNDS_UPPER))
            # Gompertz uses year offsets for numerical stability
            year_offset = years.min()
            years = years - year_offset

        # First attempt: tight bounds + low maxfev. Second attempt (still bounded):
        # higher maxfev. We never drop bounds — an unbounded fit on AI-benchmark
        # data has produced absurd parameters (e.g. inflection at 1880) silently.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                popt, pcov = curve_fit(
                    func, years, scores,
                    p0=p0, bounds=bounds,
                    maxfev=10_000,
                )
            except (RuntimeError, Warning) as e:
                logger.warning(
                    "Curve fit warning: %s. Retrying with extended maxfev (bounds kept).",
                    e,
                )
                popt, pcov = curve_fit(
                    func, years, scores,
                    p0=p0, bounds=bounds,
                    maxfev=100_000,
                )

        # Standard errors and 95% CIs via t-distribution
        std_errors = np.sqrt(np.diag(pcov))
        dof = max(len(years) - len(popt), 1)
        t_val = t_dist.ppf(0.975, dof)
        ci_95 = np.array([
            [popt[i] - t_val * std_errors[i], popt[i] + t_val * std_errors[i]]
            for i in range(len(popt))
        ])

        # Goodness of fit
        y_pred = func(years, *popt)
        ss_res = np.sum((scores - y_pred) ** 2)
        ss_tot = np.sum((scores - scores.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(ss_res / len(scores))

        # Restore Gompertz year offset
        if self.model == "gompertz":
            years = years + year_offset

        return FitResult(
            params=popt,
            param_names=param_names,
            covariance=pcov,
            std_errors=std_errors,
            ci_95=ci_95,
            r_squared=r_squared,
            rmse=rmse,
            n_points=len(years),
            model_name=self.model,
        )

    def fit_all_benchmarks(
        self, df: pd.DataFrame
    ) -> dict[str, FitResult]:
        """
        Fit curves for each benchmark in a tidy DataFrame.

        Parameters
        ----------
        df : DataFrame with columns [benchmark, year, score_norm]

        Returns
        -------
        dict mapping benchmark name → FitResult
        """
        results = {}
        for benchmark, group in df.groupby("benchmark"):
            group = group.dropna(subset=["score_norm"])
            if len(group) < 4:
                logger.warning(f"Not enough data for {benchmark}, skipping.")
                continue
            try:
                result = self.fit(group["year"].values, group["score_norm"].values)
                results[benchmark] = result
                logger.info(
                    f"{benchmark}: R²={result.r_squared:.3f}, "
                    f"inflection={result.inflection_year():.1f}, "
                    f"99% by {result.year_to_reach(0.99)}"
                )
            except Exception as e:
                logger.error(f"Failed to fit {benchmark}: {e}")
        return results
