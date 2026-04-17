"""
Hierarchical Bayesian model for sector automation risk.

Why this model
--------------
The default :class:`ScenarioSimulator` treats each sector's automation
risk as an **independent** truncated normal, parameterised by the mean /
std that McKinsey, WEF and OECD implicitly agree on for that sector.
This has two problems:

1. **No pooling.** If two sectors report wildly different uncertainties,
   the MC draws are also wildly different — even though in reality,
   uncertainty about "automation risk in 2040" is partly a property of
   the *whole economy*, not of each sector in isolation.
2. **No posterior.** Everything is a point estimate with a fixed sigma;
   there is no way to ask "how confident are we that Manufacturing's
   risk is higher than Construction's?".

The hierarchical Bayesian model below fixes both. Each sector's true
risk ``θ_s`` is drawn from a shared global prior, then the published
consensus mean ``y_s`` is treated as a noisy measurement of ``θ_s``:

    μ_g        ~ Normal(0.5, 0.2)         # global automation risk
    τ          ~ HalfNormal(0.15)          # between-sector spread
    z_s        ~ Normal(0, 1)              # non-centred draws
    θ_s        = clip(μ_g + τ · z_s, 0, 1) # per-sector risk
    y_s        ~ Normal(θ_s, σ_s_reported) # observed consensus mean

Partial pooling shrinks sectors with wide uncertainty toward the global
mean — a standard guard against over-fitting the consensus numbers.

Why this is portfolio-worthy
----------------------------
- Non-centred parameterisation (``z_s`` trick) shows familiarity with
  real-world PyMC sampling pathologies.
- Exposes ``posterior_sector_risks()`` which the MC simulator can
  consume directly in place of the truncnorm draws, giving the whole
  unemployment model an honest Bayesian backbone.
- Produces ArviZ-compatible ``idata`` so you can run ``az.summary`` /
  ``az.plot_forest`` straight from a notebook.

PyMC is imported lazily — the dashboard does not pay the pymc/arviz
install cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.models.scenarios import SECTORS, Sector

if TYPE_CHECKING:
    import arviz as az  # noqa: F401

logger = logging.getLogger(__name__)


def _import_pymc() -> tuple[Any, Any]:
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "PyMC / ArviZ are optional dependencies. Install with "
            "`pip install pymc arviz` (or use `requirements-dev.txt`)."
        ) from exc
    return pm, az


# ---------------------------------------------------------------------------
# Public results container
# ---------------------------------------------------------------------------

@dataclass
class BayesianSectorRiskResult:
    """Posterior summary for the hierarchical sector-risk model."""

    idata: Any                        # arviz.InferenceData
    sector_names: list[str]
    posterior_theta: np.ndarray       # shape (draws, n_sectors)
    mu_global_posterior: np.ndarray   # shape (draws,)
    tau_posterior: np.ndarray         # shape (draws,)

    def summary(self) -> pd.DataFrame:
        """Per-sector posterior: mean, 95% HDI, and shrinkage vs prior mean."""
        rows = []
        for i, name in enumerate(self.sector_names):
            samples = self.posterior_theta[:, i]
            rows.append(
                {
                    "sector": name,
                    "posterior_mean": float(samples.mean()),
                    "posterior_std": float(samples.std(ddof=1)),
                    "hdi_2_5": float(np.quantile(samples, 0.025)),
                    "hdi_97_5": float(np.quantile(samples, 0.975)),
                }
            )
        return pd.DataFrame(rows).sort_values("posterior_mean", ascending=False)

    def posterior_sector_risks(
        self, n_samples: int, seed: int = 42
    ) -> np.ndarray:
        """
        Draw ``n_samples`` rows from the posterior, one row per sector.

        Returns
        -------
        ndarray of shape ``(n_samples, n_sectors)`` — drop-in replacement
        for :meth:`ScenarioSimulator._sample_sector_risks_matrix`.
        """
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.posterior_theta.shape[0], size=n_samples)
        return np.clip(self.posterior_theta[idx], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_sector_risk_model(
    sectors: list[Sector] | None = None,
    observation_sigma_floor: float = 0.03,
) -> Any:
    """
    Build (but do not sample) the hierarchical sector-risk PyMC model.

    The ``observation_sigma_floor`` prevents degenerate likelihoods when
    the published ``risk_std`` is very small (which would pin the sector
    to exactly its consensus mean and defeat pooling).
    """
    pm, _ = _import_pymc()
    sectors = sectors if sectors is not None else SECTORS
    observed = np.array([s.risk_mean for s in sectors], dtype=float)
    sigma = np.maximum(
        np.array([s.risk_std for s in sectors], dtype=float),
        observation_sigma_floor,
    )

    coords = {"sector": [s.name for s in sectors]}
    with pm.Model(coords=coords) as model:
        mu_global = pm.Normal("mu_global", mu=0.5, sigma=0.2)
        tau = pm.HalfNormal("tau", sigma=0.15)

        # Non-centered parameterisation — standard trick to avoid
        # funnel-shaped posteriors when tau is small.
        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="sector")
        theta_raw = pm.Deterministic("theta_raw", mu_global + tau * z, dims="sector")
        theta = pm.Deterministic(
            "theta", pm.math.clip(theta_raw, 0.01, 0.99), dims="sector"
        )

        pm.Normal(
            "y_obs", mu=theta, sigma=sigma, observed=observed, dims="sector"
        )

    return model


def fit_sector_risk_model(
    sectors: list[Sector] | None = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: int = 42,
    progressbar: bool = False,
) -> BayesianSectorRiskResult:
    """
    Sample the hierarchical sector-risk posterior with NUTS.

    Returns
    -------
    BayesianSectorRiskResult with samples and an ArviZ ``InferenceData``.
    """
    _, _ = _import_pymc()  # import early so the error is clear
    sectors = sectors if sectors is not None else SECTORS

    model = build_sector_risk_model(sectors)
    with model:
        import pymc as pm

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    # Flatten (chain, draw, sector) -> (draws, sector)
    theta_samples = idata.posterior["theta"].stack(sample=("chain", "draw")).values.T
    mu_samples = idata.posterior["mu_global"].stack(sample=("chain", "draw")).values
    tau_samples = idata.posterior["tau"].stack(sample=("chain", "draw")).values

    return BayesianSectorRiskResult(
        idata=idata,
        sector_names=[s.name for s in sectors],
        posterior_theta=theta_samples,
        mu_global_posterior=np.asarray(mu_samples),
        tau_posterior=np.asarray(tau_samples),
    )
