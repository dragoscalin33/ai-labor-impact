"""
AI Labor Market Impact Observatory — Streamlit Dashboard.

Single page, two panels: AI capability curve (top) and sector job
displacement (bottom). The story is meant to land in twenty seconds.

Run with:
    streamlit run app/dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import streamlit as st

from src.data.ai_benchmarks import get_benchmark_dataframe, get_swe_bench_series
from src.models.benchmark_curve import BenchmarkCurveFitter
from src.models.unemployment import UnemploymentProjector
from src.viz.plots import plot_benchmark_progression, plot_sector_risk_heatmap

# ──────────────────────────────────────────────────────────────────────────────
# Page config + minimal styling
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Labor Market Impact Observatory",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        visibility: hidden;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 1000px;
    }

    /* Hero */
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 1.4rem;
    }
    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.02em;
        color: #1f2328;
        margin: 0 0 1.2rem 0;
    }
    .hero-title .accent { color: #1f6feb; }
    .hero-dek {
        font-size: 1.08rem;
        color: #57606a;
        line-height: 1.55;
        max-width: 720px;
        margin: 0 0 0.6rem 0;
    }
    .hero-rule {
        border: 0;
        border-top: 1px solid #e5e7eb;
        margin: 2.6rem 0 0 0;
    }

    /* Section */
    .section-num {
        display: inline-block;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.78rem;
        color: #1f6feb;
        background: #ddebff;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.7rem;
    }
    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1f2328;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.01em;
    }
    .section-dek {
        font-size: 0.97rem;
        color: #57606a;
        line-height: 1.55;
        max-width: 760px;
        margin: 0 0 1.1rem 0;
    }
    .section-spacer { margin-top: 3rem; }

    /* Method footer */
    .method {
        margin-top: 3.5rem;
        padding: 1.2rem 1.4rem;
        background: #f6f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        font-size: 0.86rem;
        color: #57606a;
        line-height: 1.6;
    }
    .method b { color: #1f2328; }

    .footer {
        margin-top: 1.4rem;
        font-size: 0.78rem;
        color: #6b7280;
        text-align: center;
    }
    .footer a { color: #1f6feb; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached compute
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fitting capability curve...")
def fit_benchmarks():
    df = get_benchmark_dataframe(normalize=True)
    years, scores = get_swe_bench_series()
    fitter = BenchmarkCurveFitter(model="sigmoid")
    swe_fit = fitter.fit(years, scores)
    all_fits = fitter.fit_all_benchmarks(df)
    return df, swe_fit, all_fits


@st.cache_data(ttl=3600, show_spinner="Running scenarios...")
def compute_projections():
    _, swe_fit, _ = fit_benchmarks()
    proj = UnemploymentProjector(
        years=np.arange(2025, 2051),
        n_samples=2000,
        seed=42,
        fit_result=swe_fit,
    )
    sector = proj.sector_impact(scenario="base")
    scenarios_df = proj.run_and_summarize()
    peak = proj.peak_unemployment(scenarios_df)
    return sector, peak


df_benchmarks, swe_fit, all_fits = fit_benchmarks()
sector_df, peak_df = compute_projections()

swe_obs = df_benchmarks[df_benchmarks["benchmark"] == "swe_bench"].sort_values("year")
first_obs = swe_obs.iloc[0]
mythos_obs = swe_obs[swe_obs["model"] == "Claude Mythos Preview"].iloc[0]

start_score = float(first_obs["score"])
mythos_score = float(mythos_obs["score"])
years_elapsed = float(mythos_obs["year"]) - float(first_obs["year"])

displaced_total_M = float(sector_df["displaced_jobs_M"].sum())
employed_total_M = float(sector_df["employment_2025_M"].sum())
global_displaced_pct = 100.0 * displaced_total_M / employed_total_M
top_sector_pct = float(sector_df["displacement_pct"].max())


def _peak_for(name_substr: str) -> float:
    row = peak_df[peak_df["scenario_name"].str.contains(name_substr, case=False, na=False)]
    return float(row["peak_unemployment_pct"].iloc[0]) if not row.empty else 0.0


base_peak = _peak_for("base")
opt_peak = _peak_for("optimistic")
pess_peak = _peak_for("pessimistic")

# ──────────────────────────────────────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="eyebrow">AI Labor Market Impact Observatory</div>
<h1 class="hero-title">
    By 2040, AI displaces <span class="accent">1 in 4</span> jobs worldwide.
    <br/>That's the <span class="accent">base case</span> — not the worst.
</h1>
<p class="hero-dek">
    {displaced_total_M:,.0f} million jobs gone. {top_sector_pct:.0f}% of administrative workers
    replaced. The optimistic scenario still loses {opt_peak:.0f}% of global employment;
    the pessimistic one, {pess_peak:.0f}%. Below: the AI capability curve driving the
    projection, and the sectors absorbing the impact. Every number is fitted to a
    public data source. Every projection is a Monte Carlo distribution.
</p>
<hr class="hero-rule"/>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 1 — Capability
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="section-spacer">
    <span class="section-num">01 · WHY</span>
    <h2 class="section-title">In three years, AI went from {start_score:.1f}% to {mythos_score:.1f}% on the same task</h2>
    <p class="section-dek">
        Autonomous bug resolution on SWE-bench Verified — the most-cited benchmark
        for real-world software engineering. The curve is a logistic
        <code>f(t) = L / (1 + e<sup>−k(t−t₀)</sup>)</code> fitted by bounded non-linear
        least squares on every published score. The shaded band is the 95% CI
        propagated from the parameter covariance via Monte Carlo.
    </p>
</div>
""", unsafe_allow_html=True)

proj_years = np.linspace(2020, 2032, 400)
fig_curve = plot_benchmark_progression(df_benchmarks, all_fits, proj_years, "swe_bench")
fig_curve.update_layout(height=520, margin=dict(l=60, r=120, t=70, b=60))
st.plotly_chart(fig_curve, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Panel 2 — Sector impact
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="section-spacer">
    <span class="section-num">02 · WHO</span>
    <h2 class="section-title">{top_sector_pct:.0f}% of administrative jobs gone. {global_displaced_pct:.0f}% of the world's workforce affected.</h2>
    <p class="section-dek">
        The fitted capability curve drives a vectorised Monte Carlo (n=2,000) over
        eleven sectors. Per-sector automation risk is calibrated to McKinsey, WEF
        and OECD distributions. Each bar is the share of that sector's workforce
        displaced by 2040 under the base scenario — no major policy intervention.
    </p>
</div>
""", unsafe_allow_html=True)

fig_sector = plot_sector_risk_heatmap(sector_df)
fig_sector.update_layout(height=540, margin=dict(l=60, r=120, t=70, b=60))
st.plotly_chart(fig_sector, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Methodology + footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="method">
    <b>Methodology.</b> Logistic <code>f(t) = L / (1 + exp(-k(t − t₀)))</code> fitted via
    <code>scipy.optimize.curve_fit</code> on {swe_fit.n_points} published SWE-bench Verified scores
    (R² = {swe_fit.r_squared:.3f}, RMSE = {swe_fit.rmse:.3f}). Sector displacement is a vectorised
    Monte Carlo over the joint distribution of capability-curve parameters
    (sampled from the fit covariance) and per-sector automation risks (truncated normals
    calibrated to McKinsey 2023, WEF 2023, OECD 2024). The full pipeline — including
    a hierarchical Bayesian PyMC variant of the sector model and leave-last-out
    temporal cross-validation — is reproducible end-to-end.
</div>

<div class="footer">
    Python · NumPy · SciPy · PyMC · MLflow · Streamlit · Plotly ·
    <a href="https://github.com/dragoscalin33/ai-labor-impact" target="_blank">methodology &amp; source on GitHub →</a>
</div>
""", unsafe_allow_html=True)
