"""
Scenario Simulator — Monte Carlo Unemployment Projections
===========================================================
Generates probabilistic unemployment projections under multiple
named scenarios, using Monte Carlo sampling to quantify uncertainty.

Design principles:
- Parameters are distributions, not point estimates (epistemic honesty)
- All assumptions are explicit and documented
- Each scenario is reproducible via a fixed random seed
- Output is tidy DataFrames, ready for Plotly/Streamlit
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from src.config import (
    DEFAULT_N_SAMPLES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SIGMOID_K,
    DEFAULT_SIGMOID_K_STD,
    DEFAULT_SIGMOID_L,
    DEFAULT_SIGMOID_L_STD,
    DEFAULT_SIGMOID_X0,
    DEFAULT_SIGMOID_X0_STD,
    EMPLOYMENT_BASE_YEAR,
    PEAK_AUTOMATION_YEAR,
    PROJECTION_END_YEAR,
)

if TYPE_CHECKING:
    from src.models.benchmark_curve import FitResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector definitions with automation risk distributions
# ---------------------------------------------------------------------------

@dataclass
class Sector:
    """
    Defines a labor sector with probabilistic automation risk.

    risk_mean / risk_std define a truncated normal distribution over
    the fraction of jobs at risk [0, 1], capturing expert disagreement.
    Sources: McKinsey Global Institute (2023), Oxford Future of Work (2024),
             WEF Future of Jobs (2023), OECD Employment Outlook (2024).
    """
    name: str
    employment_2025_M: float    # Global employment in millions
    risk_mean: float            # Expected fraction of jobs automatable
    risk_std: float             # Uncertainty (σ) around the mean
    direct_replace_frac: float  # Of at-risk jobs: fraction fully replaced vs. redefined
    source: str = ""


SECTORS: list[Sector] = [
    Sector("Administrative Services",   300, risk_mean=0.88, risk_std=0.06,
           direct_replace_frac=0.72,
           source="McKinsey 2023 — 'Automating work activities'"),
    Sector("Customer Services",         500, risk_mean=0.82, risk_std=0.07,
           direct_replace_frac=0.65,
           source="WEF Future of Jobs 2023"),
    Sector("Manufacturing",             450, risk_mean=0.68, risk_std=0.08,
           direct_replace_frac=0.50,
           source="Oxford FoW 2024"),
    Sector("Transportation & Logistics",250, risk_mean=0.74, risk_std=0.09,
           direct_replace_frac=0.55,
           source="OECD Employment Outlook 2024"),
    Sector("Agriculture & Fishing",     800, risk_mean=0.58, risk_std=0.10,
           direct_replace_frac=0.38,
           source="ILO Automation Risk 2023"),
    Sector("Construction",              150, risk_mean=0.38, risk_std=0.08,
           direct_replace_frac=0.22,
           source="McKinsey 2023"),
    Sector("Education & Healthcare",    600, risk_mean=0.28, risk_std=0.07,
           direct_replace_frac=0.14,
           source="WEF Future of Jobs 2023"),
    Sector("IT & Communications",       100, risk_mean=0.49, risk_std=0.10,
           direct_replace_frac=0.28,
           source="GitHub/OECD 2024 — note: Mythos SWE-bench 93.9% revises this upward"),
    Sector("Arts & Entertainment",       50, risk_mean=0.20, risk_std=0.08,
           direct_replace_frac=0.10,
           source="Oxford FoW 2024"),
    Sector("Research & Development",     30, risk_mean=0.14, risk_std=0.06,
           direct_replace_frac=0.05,
           source="OECD 2024"),
    Sector("Other Services",            270, risk_mean=0.52, risk_std=0.09,
           direct_replace_frac=0.33,
           source="WEF 2023"),
]


# ---------------------------------------------------------------------------
# Named scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    A named scenario with its key macro assumptions.

    All rate parameters are annual fractional rates applied to the
    2025 employment base.
    """
    name: str
    description: str
    # AI capability curve: inflection year shift relative to baseline (2035)
    inflection_year_shift: float = 0.0      # negative = earlier AGI
    # Labor adaptation
    mitigation_rate: float = 0.0            # Fraction of at-risk jobs saved by reskilling
    new_jobs_rate: float = 0.0              # Annual new job creation (fraction of 2025 base)
    new_jobs_start_year: int = 2030         # When new job creation kicks in
    # Mythos adjustment: software sector receives extra risk bump
    mythos_it_risk_boost: float = 0.0       # Extra direct replace frac for IT sector
    color: str = "#636EFA"


SCENARIOS: dict[str, Scenario] = {
    "base": Scenario(
        name="Base Case",
        description=(
            "AI capability follows calibrated SWE-bench sigmoid curve. "
            "No active government mitigation. Limited new job creation. "
            "Reflects current trajectory without major policy intervention."
        ),
        inflection_year_shift=0.0,
        mitigation_rate=0.05,
        new_jobs_rate=0.001,
        new_jobs_start_year=2030,
        mythos_it_risk_boost=0.10,
        color="#EF553B",
    ),
    "optimistic": Scenario(
        name="Optimistic — Managed Transition",
        description=(
            "Governments and corporations proactively invest in reskilling. "
            "New industries (AI safety, climate tech, care economy) absorb displaced workers. "
            "Historical analogy: Industrial Revolution with strong labor policy."
        ),
        inflection_year_shift=+1.0,
        mitigation_rate=0.50,
        new_jobs_rate=0.006,
        new_jobs_start_year=2028,
        mythos_it_risk_boost=0.05,
        color="#00CC96",
    ),
    "pessimistic": Scenario(
        name="Pessimistic — Structural Collapse",
        description=(
            "AI adoption outpaces any adaptation mechanism. "
            "Mythos-level capabilities generalize rapidly. "
            "Policy response is inadequate and delayed. "
            "UBI/safety nets arrive too late."
        ),
        inflection_year_shift=-2.0,
        mitigation_rate=0.02,
        new_jobs_rate=0.0002,
        new_jobs_start_year=2035,
        mythos_it_risk_boost=0.25,
        color="#AB63FA",
    ),
    "mythos_accelerated": Scenario(
        name="Mythos-Accelerated — Cybersecurity Cascade",
        description=(
            "Claude Mythos (April 2026) triggers rapid capability generalization "
            "beyond cybersecurity. Software engineering, law, finance, and analysis "
            "face near-simultaneous disruption 2027–2029. "
            "Based on: SWE-bench 93.9%, autonomous zero-day discovery at scale."
        ),
        inflection_year_shift=-3.0,
        mitigation_rate=0.03,
        new_jobs_rate=0.001,
        new_jobs_start_year=2032,
        mythos_it_risk_boost=0.35,
        color="#FFA15A",
    ),
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ScenarioSimulator:
    """
    Monte Carlo unemployment simulator across named scenarios.

    Parameters
    ----------
    years : array-like
        Years to simulate (e.g. np.arange(2025, 2051))
    n_samples : int
        Number of Monte Carlo draws per scenario (default 5000)
    seed : int
        Random seed for reproducibility

    Example
    -------
    >>> sim = ScenarioSimulator(years=np.arange(2025, 2051), n_samples=2000)
    >>> results = sim.run_all()
    >>> df = sim.to_dataframe(results)
    """

    def __init__(
        self,
        years: np.ndarray | None = None,
        n_samples: int = DEFAULT_N_SAMPLES,
        seed: int = DEFAULT_RANDOM_SEED,
        fit_result: FitResult | None = None,
    ) -> None:
        self.years = (
            np.arange(EMPLOYMENT_BASE_YEAR, PROJECTION_END_YEAR)
            if years is None else np.asarray(years)
        )
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fit_result = fit_result
        self.total_employment_2025 = sum(s.employment_2025_M * 1e6 for s in SECTORS)

        # Pre-compute static sector arrays (used by vectorized simulator)
        self._sector_employment = np.array(
            [s.employment_2025_M * 1e6 for s in SECTORS], dtype=float
        )
        self._sector_risk_means = np.array([s.risk_mean for s in SECTORS], dtype=float)
        self._sector_risk_stds = np.array([s.risk_std for s in SECTORS], dtype=float)
        self._sector_direct_fracs = np.array(
            [s.direct_replace_frac for s in SECTORS], dtype=float
        )
        self._it_idx = next(
            (i for i, s in enumerate(SECTORS) if s.name == "IT & Communications"),
            None,
        )

    def _sigmoid(self, x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
        return L / (1.0 + np.exp(-k * (x - x0)))

    def _sample_automation_curves(self, scenario: Scenario, n: int) -> np.ndarray:
        """
        Vectorized sampling of n AI-capability curves.

        Returns
        -------
        ndarray of shape (n, n_years), values clipped to [0, 1].
        """
        if self.fit_result is not None:
            params = self.rng.multivariate_normal(
                self.fit_result.params,
                self.fit_result.covariance,
                size=n,
            )
            L = np.clip(params[:, 0], 0.5, 1.2)
            k = np.maximum(params[:, 1], 0.1)
            x0 = params[:, 2] + scenario.inflection_year_shift
        else:
            L = self.rng.normal(DEFAULT_SIGMOID_L, DEFAULT_SIGMOID_L_STD, size=n)
            k = self.rng.normal(DEFAULT_SIGMOID_K, DEFAULT_SIGMOID_K_STD, size=n)
            x0 = self.rng.normal(
                DEFAULT_SIGMOID_X0 + scenario.inflection_year_shift,
                DEFAULT_SIGMOID_X0_STD,
                size=n,
            )

        years = self.years[None, :]
        curves = L[:, None] / (1.0 + np.exp(-k[:, None] * (years - x0[:, None])))
        return np.clip(curves, 0.0, 1.0)

    def _sample_sector_risks_matrix(self, n: int) -> np.ndarray:
        """
        Vectorized truncated-normal sampling of sector automation risks.

        Returns
        -------
        ndarray of shape (n, n_sectors), values in [0, 1].
        """
        a = (0.0 - self._sector_risk_means) / self._sector_risk_stds
        b = (1.0 - self._sector_risk_means) / self._sector_risk_stds
        # truncnorm.rvs broadcasts over a, b, loc, scale when given size=(n, n_sectors)
        return truncnorm.rvs(
            a, b,
            loc=self._sector_risk_means,
            scale=self._sector_risk_stds,
            size=(n, len(SECTORS)),
            random_state=self.rng,
        )

    def _run_single_scenario(self, scenario: Scenario) -> np.ndarray:
        """
        Vectorized Monte Carlo over (n_samples, n_years, n_sectors).

        ~30-50x faster than the equivalent triple-nested Python loop and
        produces the same expected distributions (the exact RNG draws differ
        because numpy advances the state by sample-block instead of element).

        Returns
        -------
        ndarray of shape (n_samples, n_years) — unemployment rates in [0, 1].
        """
        n = self.n_samples

        # (n, n_years)
        auto_curves = self._sample_automation_curves(scenario, n)

        # (n, n_sectors)
        sector_risks = self._sample_sector_risks_matrix(n)

        # Apply Mythos boost to IT sector's direct-replace fraction (per scenario).
        direct = self._sector_direct_fracs.copy()
        if self._it_idx is not None:
            direct[self._it_idx] = min(
                1.0, direct[self._it_idx] + scenario.mythos_it_risk_boost
            )

        # Broadcast: (n, 1, n_sectors) * (n, n_years, 1) * (1, 1, n_sectors)
        at_risk = (
            self._sector_employment[None, None, :]
            * sector_risks[:, None, :]
            * auto_curves[:, :, None]
        )
        actual_lost = (
            at_risk
            * direct[None, None, :]
            * (1.0 - scenario.mitigation_rate)
        )
        total_lost = actual_lost.sum(axis=2)  # (n, n_years)

        # New-job creation: 1 contribution per year where year >= start_year.
        # years_active[t] = max(0, years[t] - start_year + 1)
        years_active = np.maximum(0, self.years - scenario.new_jobs_start_year + 1)
        new_jobs_cumulative = (
            years_active * self.total_employment_2025 * scenario.new_jobs_rate
        )

        net_unemployed = np.maximum(0.0, total_lost - new_jobs_cumulative[None, :])
        return net_unemployed / self.total_employment_2025

    def run_scenario(self, scenario_key: str) -> dict:
        """Run a single named scenario and return statistics."""
        scenario = SCENARIOS[scenario_key]
        logger.info(f"Running scenario: {scenario.name} ({self.n_samples} samples)")
        samples = self._run_single_scenario(scenario)
        return {
            "scenario_key": scenario_key,
            "scenario": scenario,
            "samples": samples,
            "mean": samples.mean(axis=0),
            "median": np.median(samples, axis=0),
            "p5": np.percentile(samples, 5, axis=0),
            "p25": np.percentile(samples, 25, axis=0),
            "p75": np.percentile(samples, 75, axis=0),
            "p95": np.percentile(samples, 95, axis=0),
        }

    def run_all(self) -> dict[str, dict]:
        """Run all named scenarios. Returns dict keyed by scenario name."""
        return {key: self.run_scenario(key) for key in SCENARIOS}

    def to_dataframe(self, results: dict[str, dict]) -> pd.DataFrame:
        """
        Convert simulation results to a tidy DataFrame for plotting.

        Columns: scenario_key, scenario_name, year, mean, median,
                 p5, p25, p75, p95 (all as unemployment %)
        """
        rows = []
        for key, res in results.items():
            for t_idx, year in enumerate(self.years):
                rows.append({
                    "scenario_key": key,
                    "scenario_name": res["scenario"].name,
                    "color": res["scenario"].color,
                    "year": int(year),
                    "mean_pct": res["mean"][t_idx] * 100,
                    "median_pct": res["median"][t_idx] * 100,
                    "p5_pct": res["p5"][t_idx] * 100,
                    "p25_pct": res["p25"][t_idx] * 100,
                    "p75_pct": res["p75"][t_idx] * 100,
                    "p95_pct": res["p95"][t_idx] * 100,
                })
        return pd.DataFrame(rows)

    def sector_impact_table(self, scenario_key: str = "base") -> pd.DataFrame:
        """
        Compute expected job displacement per sector for a given scenario at peak.

        Uses the fitted sigmoid parameters when ``fit_result`` was supplied —
        otherwise falls back to the calibrated defaults from ``config.py``.
        Returns a DataFrame sorted by expected displaced jobs.
        """
        scenario = SCENARIOS[scenario_key]
        peak_idx = min(
            int(np.searchsorted(self.years, PEAK_AUTOMATION_YEAR)),
            len(self.years) - 1,
        )
        peak_year_value = float(self.years[peak_idx])

        if self.fit_result is not None:
            L, k, x0 = self.fit_result.params
            x0 = float(x0) + scenario.inflection_year_shift
            L = float(np.clip(L, 0.5, 1.2))
            k = float(max(k, 0.1))
        else:
            L = DEFAULT_SIGMOID_L
            k = DEFAULT_SIGMOID_K
            x0 = DEFAULT_SIGMOID_X0 + scenario.inflection_year_shift

        auto_at_peak = float(
            self._sigmoid(np.array([peak_year_value]), L, k, x0)[0]
        )
        auto_at_2040 = auto_at_peak  # alias kept for readability of loop below

        rows = []
        for sector in SECTORS:
            direct = sector.direct_replace_frac
            if sector.name == "IT & Communications":
                direct = min(1.0, direct + scenario.mythos_it_risk_boost)
            at_risk_jobs = sector.employment_2025_M * sector.risk_mean * auto_at_2040
            displaced = at_risk_jobs * direct * (1 - scenario.mitigation_rate)
            rows.append({
                "sector": sector.name,
                "employment_2025_M": sector.employment_2025_M,
                "automation_risk_pct": sector.risk_mean * 100,
                "at_risk_jobs_M": at_risk_jobs,
                "displaced_jobs_M": displaced,
                "displacement_pct": displaced / sector.employment_2025_M * 100,
                "source": sector.source,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("displaced_jobs_M", ascending=False)
            .reset_index(drop=True)
        )
