"""
Unemployment Projector — Wrapper with convenience methods.
Builds on ScenarioSimulator to provide high-level analysis functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import DEFAULT_RANDOM_SEED, EMPLOYMENT_BASE_YEAR, PROJECTION_END_YEAR

from .scenarios import SCENARIOS, ScenarioSimulator

if TYPE_CHECKING:
    from src.models.benchmark_curve import FitResult


class UnemploymentProjector:
    """
    High-level API for unemployment projection analysis.

    Wraps ScenarioSimulator with analysis helpers.

    Example
    -------
    >>> proj = UnemploymentProjector()
    >>> df = proj.run_and_summarize()
    >>> sector_df = proj.sector_impact(scenario="base")
    """

    def __init__(
        self,
        years: np.ndarray | None = None,
        n_samples: int = 3000,
        seed: int = DEFAULT_RANDOM_SEED,
        fit_result: FitResult | None = None,
    ) -> None:
        self.years = (
            years if years is not None
            else np.arange(EMPLOYMENT_BASE_YEAR, PROJECTION_END_YEAR)
        )
        self.sim = ScenarioSimulator(
            years=self.years,
            n_samples=n_samples,
            seed=seed,
            fit_result=fit_result,
        )
        self._results: dict | None = None

    def run_and_summarize(self, scenarios: list[str] | None = None) -> pd.DataFrame:
        """Run scenarios and return tidy summary DataFrame."""
        keys = scenarios or list(SCENARIOS.keys())
        self._results = {k: self.sim.run_scenario(k) for k in keys}
        return self.sim.to_dataframe(self._results)

    def sector_impact(self, scenario: str = "base") -> pd.DataFrame:
        """Sector-level displacement table at peak automation (2040)."""
        return self.sim.sector_impact_table(scenario)

    def peak_unemployment(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return peak median unemployment % per scenario."""
        df = df if df is not None else self.run_and_summarize()
        return (
            df.groupby("scenario_name")["median_pct"]
            .max()
            .reset_index()
            .rename(columns={"median_pct": "peak_unemployment_pct"})
            .sort_values("peak_unemployment_pct", ascending=False)
        )

    def year_crossing(self, threshold_pct: float, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the year each scenario first crosses a given unemployment threshold."""
        df = df if df is not None else self.run_and_summarize()
        rows = []
        for scenario_name, group in df.groupby("scenario_name"):
            above = group[group["median_pct"] >= threshold_pct]
            rows.append({
                "scenario_name": scenario_name,
                "threshold_pct": threshold_pct,
                "year_crossing": int(above["year"].min()) if not above.empty else None,
            })
        return pd.DataFrame(rows)
