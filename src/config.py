"""
Project-wide configuration constants.

All "magic numbers" used by the modeling code live here so they are:
- Discoverable in one place
- Easy to override for sensitivity analysis
- Documented with their data-driven justification
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------

EMPLOYMENT_BASE_YEAR: int = 2025
"""Reference year for employment denominators. World Bank latest full year."""

PROJECTION_END_YEAR: int = 2051
"""Exclusive upper bound for projection arrays (np.arange(2025, 2051))."""

PEAK_AUTOMATION_YEAR: int = 2040
"""Year used for sector-impact snapshot (~peak of fitted sigmoid + 5y buffer)."""

# ---------------------------------------------------------------------------
# Default sigmoid parameters (used as Monte Carlo prior when no fit available)
# ---------------------------------------------------------------------------
# Source: median of SWE-bench sigmoid fits computed across 2024–2026 data.
# When ScenarioSimulator receives a real `fit_result`, these are NOT used.

DEFAULT_SIGMOID_L: float = 0.98          # asymptote (max attainable score)
DEFAULT_SIGMOID_L_STD: float = 0.02
DEFAULT_SIGMOID_K: float = 0.75          # growth rate
DEFAULT_SIGMOID_K_STD: float = 0.05
DEFAULT_SIGMOID_X0: float = 2034.0       # inflection year (max growth)
DEFAULT_SIGMOID_X0_STD: float = 1.0

# ---------------------------------------------------------------------------
# Curve-fit search bounds
# ---------------------------------------------------------------------------

SIGMOID_BOUNDS_LOWER: tuple[float, float, float] = (0.01, 0.01, 2010.0)
SIGMOID_BOUNDS_UPPER: tuple[float, float, float] = (1.5,  10.0, 2045.0)

GOMPERTZ_BOUNDS_LOWER: tuple[float, float, float] = (0.01, 0.1,   0.01)
GOMPERTZ_BOUNDS_UPPER: tuple[float, float, float] = (1.5,  1000.0, 5.0)

# ---------------------------------------------------------------------------
# Monte Carlo defaults
# ---------------------------------------------------------------------------

DEFAULT_N_SAMPLES: int = 5000
DEFAULT_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2026.04.07"
"""Data snapshot identifier. Bump when SWE_BENCH_DATA / sector tables change."""

ARTIFACTS_DIR: str = "artifacts"
"""Directory where saved fit_results, seeds, and run metadata are written."""
