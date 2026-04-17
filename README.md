# AI Labor Market Impact Observatory

A reproducible, data-driven model of how the progression of AI capability
(measured against published benchmarks) translates into projected
displacement of global labour. Every parameter is fitted to a real data
source, every projection is a Monte Carlo distribution, and every run is
auditable.

[![CI](https://github.com/dragoscalin33/ai-labor-impact/actions/workflows/ci.yml/badge.svg)](https://github.com/dragoscalin33/ai-labor-impact/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-informational)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

Live dashboard: [ai-labor-impact.streamlit.app](https://ai-labor-impact.streamlit.app)
(no installation, no API key).

---

## What this is

Most AI-and-employment analyses fall into one of two patterns. They either
present invented parameters dressed up as a model, or they present a
qualitative argument with no quantification of uncertainty. This project
does neither.

| Aspect | Common approach | This project |
|---|---|---|
| AI capability | Hand-set parameters | Sigmoid fitted to published SWE-bench / HumanEval / MMLU scores via `scipy.optimize.curve_fit` |
| Employment baseline | Hard-coded estimates | World Bank Open Data API, 1991–2024 |
| Sector risk | Single-point assumptions | Truncated normals calibrated to McKinsey, WEF and OECD; optionally upgraded to a hierarchical Bayesian posterior |
| Uncertainty | Single point estimate | Monte Carlo `n = 5,000`, vectorised over `(samples × years × sectors)` |
| Validation | `R²` on the training data | Leave-last-out and rolling-origin temporal cross-validation |
| Reproducibility | "Re-run the notebook" | `FitResult` pickled with seed and data version; every run loggable to MLflow |

---

## Quickstart

```bash
git clone https://github.com/dragoscalin33/ai-labor-impact.git
cd ai-labor-impact
pip install -r requirements.txt
streamlit run app/dashboard.py            # http://localhost:8501
```

For the Bayesian and tracking notebooks:

```bash
pip install -r requirements-dev.txt
jupyter lab notebooks/
mlflow ui                                  # http://localhost:5000
```

Or with Docker:

```bash
docker compose up --build
```

The dashboard works without any LLM API key: a curated narrative cache
(`data/insights_cache.json`) provides per-scenario commentary out of the
box. Adding a free [Groq](https://console.groq.com) key in the sidebar
swaps the cache for live LLM generation; it is never required.

---

## Methodology

### AI capability curve

A logistic function `f(t) = L / (1 + exp(-k(t - t₀)))` is fitted to
historical benchmark scores using bounded non-linear least squares
(Levenberg–Marquardt). Confidence intervals are derived from the
parameter covariance matrix via the Student-t distribution. Predictive
uncertainty is propagated by sampling `(L, k, t₀)` from the multivariate
normal `(μ, Σ)` of the fit — never by re-fitting on perturbed data.

The fit covers SWE-bench Verified (the project's primary anchor),
HumanEval and MMLU. The most recent SWE-bench observation is the
[Claude Mythos Preview](https://red.anthropic.com/2026/mythos-preview/)
report (April 2026), which scored 93.9 % — a 21-percentage-point gain
over Claude Opus 4.6 (72.5 %) in approximately fourteen months.

Source: `src/models/benchmark_curve.py`.

### Monte Carlo unemployment simulator

For each of `n = 5,000` trials:

1. Sample the sigmoid parameters from the fit covariance.
2. Sample sector automation risks from per-sector truncated normals
   (or, optionally, from the hierarchical Bayesian posterior — see
   below).
3. Apply `automation × risk × direct-replace × (1 − mitigation)` per
   sector, per year.
4. Subtract the accumulated new-job creation specific to the scenario.
5. Return an `(n_samples, n_years)` matrix of unemployment rates.

The whole loop is a single NumPy broadcast over
`(n_samples, n_years, n_sectors)`. For `n = 5,000` over twenty-six years
and eleven sectors it completes in roughly twenty-six milliseconds on a
laptop — three orders of magnitude faster than the equivalent
triple-nested Python loop.

Source: `src/models/scenarios.py`.

### Hierarchical Bayesian sector risk

The default truncated-normal sector model treats every sector as
independent. The hierarchical PyMC model in `src/models/bayesian.py`
partially pools sector estimates toward a shared global mean with a
sector-level deviation, using a non-centred parameterisation:

```
μ_g  ~ Normal(0.5, 0.2)
τ    ~ HalfNormal(0.15)
z_s  ~ Normal(0, 1)
θ_s  = clip(μ_g + τ · z_s, 0.01, 0.99)
y_s  ~ Normal(θ_s, σ_s_reported)
```

The posterior is a drop-in replacement for the truncated-normal draws in
the Monte Carlo simulator. Convergence diagnostics, forest plots, and a
prior-versus-posterior comparison are in
`notebooks/04_bayesian_sector_risk.ipynb`.

### Temporal cross-validation

Reporting `R² ≈ 0.97` on the training data is not evidence that the
sigmoid is predictive. The honest test is out-of-sample: hide the latest
observation, fit on what was available before it, and check whether the
resulting forecast contains the held-out point.

Two protocols are implemented in `src/validation/temporal_cv.py`:

- `leave_last_out` — for each observation `i`, fit on points `0..i−1`
  and predict `i`. Reports per-fold prediction, 95 % CI, and absolute
  error.
- `rolling_origin_cv` — at each cutoff date, forecast `h` years ahead
  (`h ∈ {0.5, 1.0, 2.0}`) and compare against any observation that fell
  in the window. Standard discipline from operational time-series
  forecasting (Hyndman & Athanasopoulos, *FPP3*, ch. 5).

On the SWE-bench series, the model fitted exclusively on data published
before April 2026 placed Claude Mythos Preview within its 95 % CI. The
notebook `notebooks/05_temporal_validation.ipynb` reproduces the result
end-to-end.

### Reproducibility and tracking

Every fit and every Monte Carlo run can be logged to MLflow with a single
context manager (`track_run("name")`). Tracked parameters include data
version, seed, `n_samples`, and the sigmoid coefficients; tracked
metrics include `R²`, RMSE, the inflection year, and per-scenario peak
unemployment. The full `FitResult` is logged as a pickled artifact.

For environments without MLflow, `src/persistence.py` provides the same
guarantees through plain pickle and JSON files in `artifacts/`.

---

## Repository layout

```
ai-labor-impact/
├── src/
│   ├── config.py              project-wide constants (no magic numbers)
│   ├── persistence.py         reproducibility artefacts (pickle + JSON)
│   ├── data/                  real-world API clients
│   │   ├── ai_benchmarks.py   SWE-bench, HumanEval, MMLU, plus Mythos
│   │   ├── world_bank.py      Unemployment 1991–2024 (no auth)
│   │   └── metaculus.py       Crowd-aggregated AGI forecasts
│   ├── models/
│   │   ├── benchmark_curve.py sigmoid / Gompertz fitter with CIs
│   │   ├── scenarios.py       vectorised Monte Carlo simulator
│   │   ├── unemployment.py    high-level projection API
│   │   └── bayesian.py        hierarchical PyMC model (lazy import)
│   ├── validation/
│   │   └── temporal_cv.py     leave-last-out + rolling-origin CV
│   ├── tracking/
│   │   └── mlflow_tracker.py  run logging (lazy import)
│   ├── viz/plots.py           Plotly figures (dark theme)
│   └── insights/generator.py  LLM provider abstraction + cached narrative
├── app/dashboard.py           Streamlit interactive dashboard
├── notebooks/
│   ├── 02_ai_progress_curve.ipynb
│   ├── 03_employment_impact.ipynb
│   ├── 04_bayesian_sector_risk.ipynb
│   └── 05_temporal_validation.ipynb
├── data/
│   └── insights_cache.json    bundled per-scenario commentary
├── tests/                     pytest, 18 tests
├── .github/workflows/ci.yml   ruff + mypy + pytest on 3.10 / 3.11 / 3.12
├── Dockerfile, docker-compose.yml
├── DEPLOY.md
└── requirements*.txt          tiered dependencies
```

---

## Data sources

| Source | Coverage | Authentication |
|---|---|---|
| [World Bank Open Data](https://data.worldbank.org) | Global unemployment and employment, 1991–2024 | None |
| [Papers with Code](https://paperswithcode.com) | AI benchmark scores (SWE-bench, HumanEval, MMLU) | None |
| [Metaculus](https://metaculus.com) | Crowd-aggregated AGI probability forecasts | None |
| [Anthropic red team](https://red.anthropic.com/2026/mythos-preview/) | Mythos Preview SWE-bench 93.9 % | Public report |
| McKinsey Global Institute, WEF, OECD | Sector automation risk estimates | Published reports |

---

## Tests, lint, types

```bash
pytest tests/                  # 18 passed
ruff check src/ tests/         # all checks passed
mypy src/                      # success: no issues
```

The same three checks run on every push and pull request, against Python
3.10, 3.11, and 3.12.

---

## License

MIT. See [LICENSE](LICENSE).
