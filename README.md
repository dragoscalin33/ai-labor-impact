<div align="center">

# 🤖 AI Labor Market Impact Observatory

**A rigorous, reproducible, fully-Bayesian analysis of how AI capability progression affects global employment.**

*All projections fit on real data. All uncertainty is quantified. All runs are tracked.*

[![Live Demo](https://img.shields.io/badge/⭐%20Live%20demo-Streamlit-FF4B4B?style=for-the-badge)](https://ai-labor-impact.streamlit.app)
[![CI](https://github.com/dragoscalin33/ai-labor-impact/actions/workflows/ci.yml/badge.svg)](https://github.com/dragoscalin33/ai-labor-impact/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-18%20passing-3fb950)](tests/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[**Live demo**](https://ai-labor-impact.streamlit.app) · [**Methodology**](#-methodology) · [**Notebooks**](notebooks/) · [**Deploy guide**](DEPLOY.md)

</div>

---

## ⚡ TL;DR for hiring managers

This repo is a one-stop demonstration of the ML-engineering work I do day-to-day:

| What it shows | Where to look |
|---|---|
| Vectorised NumPy — replaced a triple-nested Python loop with a single broadcast (≈ **30 s → 26 ms**, > 1000× faster) | [`src/models/scenarios.py`](src/models/scenarios.py) — `_run_single_scenario` |
| Hierarchical Bayesian model in PyMC with non-centred parameterisation | [`src/models/bayesian.py`](src/models/bayesian.py) + [notebook 04](notebooks/04_bayesian_sector_risk.ipynb) |
| Honest model validation — leave-last-out temporal CV that **predicted Mythos before April 2026** | [`src/validation/temporal_cv.py`](src/validation/temporal_cv.py) + [notebook 05](notebooks/05_temporal_validation.ipynb) |
| Experiment tracking — every fit + Monte Carlo run logged to MLflow | [`src/tracking/`](src/tracking/) |
| Reproducibility artefacts — pickled fits + JSON metadata + data version | [`src/persistence.py`](src/persistence.py) |
| Production-ready packaging — Docker, lean Streamlit Cloud deploy, tiered requirements files | [`Dockerfile`](Dockerfile), [`DEPLOY.md`](DEPLOY.md) |
| CI on Python 3.10 / 3.11 / 3.12 — ruff + mypy + pytest, all green | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

**Open to ML / ML-Eng roles.** Reach out via [LinkedIn](https://www.linkedin.com/in/dragoscalin) — I respond within a day.

---

## 🧭 What this project is

Most AI-vs-employment analyses fall into one of two traps. Either they
present "simulated" data dressed up as a model, or they make qualitative
arguments without quantifying uncertainty. This project does **neither**.

| Aspect | Common approach | This project |
|---|---|---|
| AI capability data | Invented parameters | Sigmoid fitted to real benchmark scores (Papers with Code) |
| Employment data | Hardcoded estimates | World Bank Open Data API (1991–2024) |
| Sector risk | Single-point assumptions | Truncated-normal distributions calibrated to McKinsey/WEF/OECD, optionally upgraded to a hierarchical Bayesian posterior |
| Uncertainty | Single point estimate | Monte Carlo n = 5,000, vectorised in NumPy |
| Validation | Reports R² on training data | Leave-last-out temporal CV — Mythos Preview correctly inside the 95 % CI |
| Reproducibility | "Run the notebook again" | Pickled fits + seeds + data version, every run MLflow-tracked |

---

## 🚀 Try it in 60 seconds

> **Easiest path:** click the **[Live demo](https://ai-labor-impact.streamlit.app)** badge above. No install, no API key.

Or run locally:

```bash
git clone https://github.com/dragoscalin33/ai-labor-impact.git
cd ai-labor-impact
pip install -r requirements.txt          # ~30 s — lean install
streamlit run app/dashboard.py           # → http://localhost:8501
```

Or with Docker:

```bash
docker compose up --build
# → http://localhost:8501
```

For the Bayesian / MLflow notebooks:

```bash
pip install -r requirements-dev.txt      # adds PyMC + MLflow + Jupyter
jupyter lab notebooks/
```

> **No API keys required.** The "Insights" tab uses a bundled, hand-curated narrative cache (`data/insights_cache.json`). If you *do* have a free [Groq](https://console.groq.com) key or an Anthropic key, paste it in the sidebar for live LLM commentary — but it's never required.

---

## 📊 Headline results

The SWE-bench sigmoid (R² ≈ 0.97) places the **inflection year of AI software-engineering capability between 2024 and 2026**, with the Claude Mythos Preview (93.9 %, April 2026) anchoring the rightmost data point. Four named scenarios, each n = 5,000 Monte Carlo:

| Scenario | Peak unemployment (P25–P75) | Time to 10 % | Policy posture |
|---|---|---|---|
| **Optimistic** — managed transition | 12 – 18 % | mid-2030s | Aggressive reskilling + new-industry investment |
| **Base** — no major intervention | 28 – 38 % | ~2030 | Status-quo policy |
| **Pessimistic** — structural collapse | 45 – 58 % | ~2028 | Minimal response |
| **Mythos-Accelerated** — cybersecurity cascade | 35 – 52 % | 2027 – 2028 | Crisis-level required from 2027 |

The Mythos-Accelerated scenario is calibrated to the [Mythos Preview report](https://red.anthropic.com/2026/mythos-preview/) — autonomous zero-day discovery at scale generalises to other knowledge-work domains within ~12 months.

---

## 🏗️ Architecture

```
ai-labor-impact/
├── src/
│   ├── config.py              ← project-wide constants (no magic numbers)
│   ├── persistence.py         ← reproducibility artefacts (pickle + JSON)
│   ├── data/                  ← real-world API clients
│   │   ├── ai_benchmarks.py   ← SWE-bench / HumanEval / MMLU + Mythos
│   │   ├── world_bank.py      ← Unemployment 1991-2024 (no auth)
│   │   └── metaculus.py       ← Crowd-aggregated AGI forecasts
│   ├── models/
│   │   ├── benchmark_curve.py ← Sigmoid / Gompertz fitter w/ CIs
│   │   ├── scenarios.py       ← VECTORISED Monte Carlo simulator
│   │   ├── unemployment.py    ← High-level projection API
│   │   └── bayesian.py        ← Hierarchical PyMC model (lazy import)
│   ├── validation/
│   │   └── temporal_cv.py     ← Leave-last-out + rolling-origin CV
│   ├── tracking/
│   │   └── mlflow_tracker.py  ← Run logging (lazy import)
│   ├── viz/plots.py           ← Plotly dark-theme figures
│   └── insights/generator.py  ← LLM provider abstraction + cached narrative
├── app/dashboard.py           ← Streamlit interactive dashboard
├── notebooks/
│   ├── 02_ai_progress_curve.ipynb     ← flagship: capability curve
│   ├── 03_employment_impact.ipynb     ← Monte Carlo projections
│   ├── 04_bayesian_sector_risk.ipynb  ← Bayesian sector model
│   └── 05_temporal_validation.ipynb   ← Mythos-validation notebook
├── data/
│   └── insights_cache.json    ← Pre-generated narrative (no API key needed)
├── tests/                     ← pytest, 18 tests, all passing
├── .github/workflows/ci.yml   ← ruff + mypy + pytest on 3.10/3.11/3.12
├── Dockerfile + docker-compose.yml
├── DEPLOY.md                  ← Streamlit Cloud / Docker / Local
└── requirements*.txt          ← Tiered dependencies (lean → dev)
```

---

## 🧠 Methodology

### 1 · AI capability curve

A sigmoid `f(t) = L / (1 + exp(-k(t - t₀)))` is fitted to historical SWE-bench
scores using `scipy.optimize.curve_fit` (Levenberg–Marquardt) with **bounded
parameters** so the fallback path can never silently return absurd
parameters. Confidence intervals come from the parameter covariance matrix
via the t-distribution.

Uncertainty is propagated by sampling parameters from the multivariate
normal `(μ, Σ)` of the fit — no point-estimate hand-waving.

### 2 · Monte Carlo unemployment simulator

For each of n = 5,000 trials:

1. Sample `(L, k, x₀)` from the sigmoid-fit covariance.
2. Sample sector automation risk from truncated normals (or, optionally,
   from the hierarchical Bayesian posterior — see § 4).
3. Apply `automation × risk × direct-replace × (1 − mitigation)` per sector,
   per year.
4. Subtract accumulated new-job creation.
5. Return `(n_samples, n_years)` unemployment matrix.

The whole loop is **vectorised** to a single NumPy broadcast over
`(n_samples, n_years, n_sectors)`. n = 5,000 over 26 years × 11 sectors
runs in **≈ 26 ms** on a laptop — > 1000× faster than the original
triple-nested Python loop.

### 3 · Temporal cross-validation

Rather than reporting `R² = 0.97` on the training data and calling it a
day, the project includes **leave-last-out** validation: refit the
sigmoid on every prefix of the data and predict the held-out
observation. The Mythos Preview (April 2026) lands inside the 95 % CI of
a model fit on data available *before it dropped*.

This is the discipline used in real time-series forecasting
(Hyndman & Athanasopoulos, *FPP3* ch. 5) — see
[`notebooks/05_temporal_validation.ipynb`](notebooks/05_temporal_validation.ipynb).

### 4 · Hierarchical Bayesian sector risk

Optional but present: instead of treating each sector's automation risk
as an independent truncated normal, the PyMC model in
[`src/models/bayesian.py`](src/models/bayesian.py) **partially pools**
sectors toward a global mean with a shared between-sector spread `τ`,
using non-centred parameterisation to avoid Neal's funnel. The posterior
is a drop-in replacement for the Monte Carlo sector draws.

### 5 · Reproducibility & tracking

Every fit + scenario run is loggable to MLflow with a single context
manager (`track_run("name")`). Tracked params include data version, seed,
n_samples, sigmoid parameters; tracked metrics include R², RMSE,
inflection year, and per-scenario peak unemployment. Artefacts include
the full pickled `FitResult` and a human-readable JSON manifest.

`src/persistence.py` provides MLflow-independent `save_run` / `load_run`
helpers so the project remains reproducible even on a minimal install.

---

## 🔌 Data sources

| Source | Data | Auth |
|---|---|---|
| [World Bank Open Data](https://data.worldbank.org) | Global unemployment & employment 1991-2024 | None |
| [Papers with Code](https://paperswithcode.com) | AI benchmark scores (SWE-bench, HumanEval, MMLU) | None |
| [Metaculus](https://metaculus.com) | Crowd-aggregated AGI probability forecasts | None |
| [Anthropic red team](https://red.anthropic.com/2026/mythos-preview/) | Mythos Preview benchmark (SWE-bench 93.9 %) | Public |
| McKinsey Global Institute, WEF, OECD | Sector automation risk estimates | Published reports |

---

## 🧪 Tests, lint, types

```bash
pytest tests/                 # 18 passed in <2 s
ruff check src/ tests/        # All checks passed!
mypy src/                     # Type-clean
```

CI runs the same checks on Python 3.10, 3.11, and 3.12 on every push and PR.

---

## 📈 Context: Claude Mythos Preview (April 2026)

On April 7 2026 Anthropic published the [Mythos Preview report](https://red.anthropic.com/2026/mythos-preview/). The data points used here:

- **SWE-bench Verified: 93.9 %** (+21 pp vs. Claude Opus 4.6 at 72.5 % in ~14 months)
- Autonomous discovery of thousands of zero-day vulnerabilities including a 17-year-old FreeBSD RCE
- Restricted to [Project Glasswing](https://www.anthropic.com/glasswing) partners

This single observation places AI software-engineering capability firmly
in the *expert human* range and is the rightmost anchor of the SWE-bench
sigmoid. Everything else in the dashboard is downstream of this fit.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

*Built by [Dragos](https://github.com/dragoscalin33) — open to ML / ML-Engineering roles.
[LinkedIn](https://www.linkedin.com/in/dragoscalin) · [Live demo](https://ai-labor-impact.streamlit.app)*

</div>
