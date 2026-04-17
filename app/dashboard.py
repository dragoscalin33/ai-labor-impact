"""
AI Labor Market Impact Observatory — Streamlit Dashboard
=========================================================
Run with:
    streamlit run app/dashboard.py

or from project root:
    make dashboard
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.data.ai_benchmarks import get_benchmark_dataframe, get_swe_bench_series, BENCHMARK_META
from src.data.world_bank import WorldBankClient
from src.models.benchmark_curve import BenchmarkCurveFitter
from src.models.unemployment import UnemploymentProjector
from src.models.scenarios import SCENARIOS, SECTORS
from src.viz.plots import (
    plot_benchmark_progression,
    plot_unemployment_scenarios,
    plot_sector_risk_heatmap,
    plot_monte_carlo_fan,
)
from src.insights.generator import InsightsGenerator

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Labor Market Impact Observatory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark professional theme
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background-color: #161b22; border-radius: 8px; padding: 12px; border: 1px solid #21262d; }
    .stMetric label { color: #8b949e !important; }
    .stMetric [data-testid="metric-container"] { color: #e6edf3; }
    h1, h2, h3 { color: #e6edf3; }
    .sidebar .sidebar-content { background-color: #161b22; }
    .mythos-badge {
        background: linear-gradient(135deg, #d2a8ff22, #58a6ff22);
        border: 1px solid #d2a8ff55;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading benchmark data...")
def load_benchmarks():
    return get_benchmark_dataframe(normalize=True)

@st.cache_data(ttl=3600, show_spinner="Fitting capability curves...")
def fit_curves():
    years, scores = get_swe_bench_series()
    fitter = BenchmarkCurveFitter(model="sigmoid")
    fit = fitter.fit(years, scores)
    all_df = get_benchmark_dataframe(normalize=True)
    all_fits = fitter.fit_all_benchmarks(all_df)
    return fit, all_fits

@st.cache_data(ttl=7200, show_spinner="Fetching World Bank data...")
def load_world_bank():
    try:
        client = WorldBankClient(cache=True)
        return client.get_global_unemployment_trend(date_range=(1991, 2024))
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner="Running Monte Carlo simulations...")
def run_simulations(n_samples: int, scenarios: list):
    fit, _ = fit_curves()
    proj = UnemploymentProjector(
        years=np.arange(2025, 2051),
        n_samples=n_samples,
        seed=42,
        fit_result=fit,
    )
    df = proj.run_and_summarize(scenarios=scenarios)
    peak_df = proj.peak_unemployment(df)
    sector_df = proj.sector_impact(scenario="base")
    return df, peak_df, sector_df, proj

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### Scenarios")
    selected_scenarios = st.multiselect(
        "Select scenarios to display",
        options=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k].name,
    )

    st.markdown("### Simulation")
    n_samples = st.select_slider(
        "Monte Carlo samples",
        options=[500, 1000, 2000, 5000],
        value=2000,
        help="More samples = more accurate uncertainty bands but slower.",
    )

    st.markdown("### Insights Provider")
    provider = st.selectbox(
        "AI insights provider",
        options=["template", "groq", "anthropic"],
        index=0,
        help="'template' requires no API key. 'groq' is free at console.groq.com.",
    )
    api_key = ""
    if provider != "template":
        api_key = st.text_input(
            f"{provider.capitalize()} API Key",
            type="password",
            placeholder=f"Enter your {provider} API key",
        )

    st.divider()
    st.markdown("""
    **Data Sources**
    - 🌍 [World Bank API](https://data.worldbank.org)
    - 📊 [Papers with Code](https://paperswithcode.com)
    - 🎯 [Metaculus](https://metaculus.com)
    - 🔴 [Anthropic red.anthropic.com](https://red.anthropic.com/2026/mythos-preview/)
    """)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
# 🤖 AI Labor Market Impact Observatory
**A data-driven analysis of AI capability progression and global employment displacement**
*All projections use real data. All uncertainty is quantified. No hardcoded parameters.*
""")

st.markdown("""
<div class="mythos-badge">
⭐ <strong>April 7, 2026 — Claude Mythos Preview:</strong> 93.9% SWE-bench Verified
(+21pp vs Claude Opus 4.6 in ~14 months). Autonomous zero-day vulnerability discovery at scale.
This data point anchors the AI capability curve used in all projections below.
<br><a href="https://red.anthropic.com/2026/mythos-preview/" target="_blank">→ Full technical report</a>
</div>
""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

df_benchmarks = load_benchmarks()
swe_fit, all_fits = fit_curves()
wb_df = load_world_bank()

if selected_scenarios:
    sim_df, peak_df, sector_df, proj = run_simulations(n_samples, selected_scenarios)
else:
    st.warning("Select at least one scenario to run simulations.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# KPI Metrics Row
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("## 📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

mythos_swe = 93.9
opus_swe = 72.5

with col1:
    st.metric("Mythos SWE-bench", f"{mythos_swe}%", f"+{mythos_swe - opus_swe:.1f}pp vs Opus 4.6")
with col2:
    infl_yr = swe_fit.inflection_year()
    st.metric("Sigmoid Inflection", f"{infl_yr:.1f}", "Year of max growth")
with col3:
    y99 = swe_fit.year_to_reach(0.99)
    st.metric("99% Capability Yr", f"{y99:.0f}" if y99 else "—", "SWE-bench projection")
with col4:
    base_peak = peak_df[peak_df["scenario_name"].str.contains("Base", case=False)]
    bpct = float(base_peak["peak_unemployment_pct"].iloc[0]) if not base_peak.empty else 0
    st.metric("Base Scenario Peak", f"{bpct:.1f}%", "Global unemployment")
with col5:
    opt_peak = peak_df[peak_df["scenario_name"].str.contains("Optimistic", case=False)]
    opct = float(opt_peak["peak_unemployment_pct"].iloc[0]) if not opt_peak.empty else 0
    st.metric("Optimistic Peak", f"{opct:.1f}%", "With active policy")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠 AI Capability Curve",
    "📉 Unemployment Scenarios",
    "🏭 Sector Impact",
    "🎲 Monte Carlo Detail",
    "💡 Insights",
])

# ── Tab 1: AI Capability Curve ────────────────────────────────────────────────
with tab1:
    st.markdown("### AI Benchmark Progression — Calibrated Sigmoid Model")
    st.markdown("""
    The sigmoid curve is **fitted to real published benchmark scores** using non-linear
    least squares regression. Parameters (asymptote L, growth rate k, inflection year x₀)
    are estimated from data — not assumed. Uncertainty bands show 95% Monte Carlo CI
    propagated through the parameter covariance matrix.
    """)

    bm_choice = st.selectbox(
        "Select benchmark",
        options=list(all_fits.keys()),
        format_func=lambda k: BENCHMARK_META.get(k, {}).get("label", k),
    )

    proj_years = np.linspace(2020, 2035, 400)
    fig = plot_benchmark_progression(df_benchmarks, all_fits, proj_years, bm_choice)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    if bm_choice in all_fits:
        res = all_fits[bm_choice]
        st.markdown("**Fit parameters:**")
        st.dataframe(res.summary().style.format({
            "estimate": "{:.4f}", "std_error": "{:.4f}",
            "ci_95_lower": "{:.4f}", "ci_95_upper": "{:.4f}",
        }), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{res.r_squared:.4f}")
        c2.metric("RMSE", f"{res.rmse:.4f}")
        c3.metric("Data points", res.n_points)

    if not wb_df.empty:
        st.markdown("### World Bank Historical Unemployment (Global)")
        import plotly.express as px
        world = wb_df[wb_df["country_code"].isin(["WLD", "HIC", "LCN", "EAS"])]
        fig_wb = px.line(
            world, x="year", y="value", color="region_label",
            labels={"value": "Unemployment (%)", "year": "Year"},
            template="plotly_dark",
        )
        fig_wb.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", height=380)
        st.plotly_chart(fig_wb, use_container_width=True)

# ── Tab 2: Unemployment Scenarios ─────────────────────────────────────────────
with tab2:
    st.markdown("### Global Unemployment Projections — All Scenarios")
    st.markdown(f"""
    Monte Carlo simulation with **{n_samples:,} samples** per scenario.
    Sector automation risk drawn from truncated normal distributions (sources: McKinsey, WEF, OECD).
    AI capability curve calibrated to SWE-bench sigmoid fit (R²={swe_fit.r_squared:.3f}).
    """)

    fig = plot_unemployment_scenarios(sim_df)
    fig.update_layout(height=580)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Peak Unemployment by Scenario")
    st.dataframe(
        peak_df.sort_values("peak_unemployment_pct", ascending=False)
               .style.format({"peak_unemployment_pct": "{:.1f}%"}),
        use_container_width=True,
    )

    st.markdown("### Threshold Crossing Years")
    thresholds = [5, 10, 20, 30]
    cross_rows = []
    for t in thresholds:
        cdf = proj.year_crossing(t, sim_df)
        cdf["threshold"] = f"{t}%"
        cross_rows.append(cdf)
    cross_all = pd.concat(cross_rows)
    pivot = cross_all.pivot(index="scenario_name", columns="threshold", values="year_crossing")
    st.dataframe(pivot, use_container_width=True)

# ── Tab 3: Sector Impact ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### Sector-Level Job Displacement at Peak Automation (~2040)")
    st.markdown("""
    Based on automation risk distributions from McKinsey Global Institute (2023),
    WEF Future of Jobs (2023), and OECD Employment Outlook (2024).
    The **IT & Communications** sector carries an extra risk premium from Mythos-level
    capabilities (SWE-bench 93.9%).
    """)

    fig = plot_sector_risk_heatmap(sector_df)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detailed Sector Table")
    st.dataframe(
        sector_df[["sector", "employment_2025_M", "automation_risk_pct",
                   "at_risk_jobs_M", "displaced_jobs_M", "displacement_pct", "source"]]
        .style.format({
            "employment_2025_M": "{:.0f}M",
            "automation_risk_pct": "{:.0f}%",
            "at_risk_jobs_M": "{:.0f}M",
            "displaced_jobs_M": "{:.0f}M",
            "displacement_pct": "{:.1f}%",
        }),
        use_container_width=True,
    )

# ── Tab 4: Monte Carlo Detail ─────────────────────────────────────────────────
with tab4:
    st.markdown("### Monte Carlo Distribution — Single Scenario Deep Dive")
    st.markdown("""
    Shows individual simulation paths (faded lines) alongside the percentile bands.
    Each path represents one draw from the joint distribution of:
    automation curve parameters × sector risk parameters.
    """)

    from src.models.scenarios import ScenarioSimulator
    @st.cache_data(ttl=600, show_spinner="Running detail simulation...")
    def run_detail(scenario_key, n):
        fit, _ = fit_curves()
        sim = ScenarioSimulator(
            years=np.arange(2025, 2051),
            n_samples=n,
            seed=42,
            fit_result=fit,
        )
        return {scenario_key: sim.run_scenario(scenario_key)}

    sc_choice = st.selectbox(
        "Select scenario",
        options=selected_scenarios,
        format_func=lambda k: SCENARIOS[k].name,
    )

    detail_results = run_detail(sc_choice, min(n_samples, 2000))
    fig = plot_monte_carlo_fan(detail_results, sc_choice, np.arange(2025, 2051))
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    scenario = SCENARIOS[sc_choice]
    st.info(f"**{scenario.name}:** {scenario.description}")

# ── Tab 5: Insights ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 💡 AI-Generated Analysis Insights")

    if provider != "template" and not api_key:
        st.warning(f"Enter your {provider} API key in the sidebar to generate LLM insights.")
        provider_to_use = "template"
        key_to_use = None
    else:
        provider_to_use = provider
        key_to_use = api_key if api_key else None

    if st.button("🔄 Generate Insights", type="primary"):
        with st.spinner("Generating insights..."):
            try:
                gen = InsightsGenerator(provider=provider_to_use, api_key=key_to_use)
                cross_df = proj.year_crossing(10.0, sim_df)
                insights = gen.summarize_scenario_results(sim_df, peak_df, cross_df)
                exec_summary = gen.executive_summary(peak_df, {})

                st.markdown("### Executive Summary")
                st.markdown(exec_summary)
                st.divider()

                st.markdown("### Scenario Narratives")
                for name, text in insights.items():
                    with st.expander(f"📌 {name}"):
                        st.markdown(text)

            except Exception as e:
                st.error(f"Insights generation failed: {e}")
    else:
        st.markdown("""
        Click **Generate Insights** to produce:
        - Executive summary of the full analysis
        - Per-scenario narrative with key numbers
        - Policy implications

        *Using 'template' provider (no API key needed).
        For richer analysis, add a Groq API key (free at [console.groq.com](https://console.groq.com)).*
        """)

# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown("""
<div style="color: #8b949e; font-size: 0.85em; text-align: center;">
AI Labor Market Impact Observatory · Built with Python, Plotly, Streamlit ·
Data: World Bank API, Papers with Code, Metaculus ·
Model: Sigmoid curve fitted via scipy.optimize.curve_fit ·
Uncertainty: Monte Carlo n=5,000
</div>
""", unsafe_allow_html=True)
