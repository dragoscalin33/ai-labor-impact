"""
Professional Plotly Visualizations
=====================================
All charts use a consistent dark theme with Anthropic-inspired accent colors.
Each function returns a Plotly Figure — renderable in notebooks, Streamlit,
or saved as HTML/PNG/SVG.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

THEME = {
    "bg": "#0d1117",
    "paper": "#161b22",
    "grid": "#21262d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "accent": "#58a6ff",
    "accent2": "#f78166",
    "accent3": "#3fb950",
    "mythos": "#d2a8ff",   # Purple for Mythos data point
}

FONT = dict(family="Inter, -apple-system, sans-serif", color=THEME["text"])

BASE_LAYOUT = dict(
    paper_bgcolor=THEME["paper"],
    plot_bgcolor=THEME["bg"],
    font=FONT,
    xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"]),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor=THEME["grid"],
        borderwidth=1,
    ),
    margin=dict(l=60, r=30, t=70, b=60),
)


def _apply_theme(fig: go.Figure, title: str = "", subtitle: str = "") -> go.Figure:
    full_title = f"<b>{title}</b>" if not subtitle else f"<b>{title}</b><br><sup>{subtitle}</sup>"
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=full_title, font=dict(size=18), x=0.0, xanchor="left"),
    )
    return fig


# ---------------------------------------------------------------------------
# 1. AI Benchmark Progression
# ---------------------------------------------------------------------------

def plot_benchmark_progression(
    df: pd.DataFrame,
    fit_results: dict | None = None,
    projection_years: np.ndarray | None = None,
    benchmark: str = "swe_bench",
) -> go.Figure:
    """
    Plot historical AI benchmark scores with fitted sigmoid curve and
    95% uncertainty band.

    The Mythos Preview data point is highlighted distinctly.
    """
    data = df[df["benchmark"] == benchmark].sort_values("year")
    proj_years = projection_years if projection_years is not None else np.linspace(
        data["year"].min(), 2035, 300
    )

    fig = go.Figure()

    # --- Uncertainty band (if fit available) ---
    if fit_results and benchmark in fit_results:
        fit = fit_results[benchmark]
        mean, lower, upper = fit.predict_with_uncertainty(proj_years)

        fig.add_trace(go.Scatter(
            x=np.concatenate([proj_years, proj_years[::-1]]),
            y=np.concatenate([upper * 100, (lower * 100)[::-1]]),
            fill="toself",
            fillcolor="rgba(88,166,255,0.12)",
            line=dict(width=0),
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=proj_years,
            y=mean * 100,
            mode="lines",
            line=dict(color=THEME["accent"], width=2.5, dash="dash"),
            name="Fitted sigmoid (projected)",
        ))

        # Inflection year annotation
        infl = fit.inflection_year()
        fig.add_vline(
            x=infl, line_dash="dot", line_color=THEME["accent"],
            annotation_text=f"Inflection ~{infl:.0f}",
            annotation_font_color=THEME["accent"],
        )

    # --- Observed data points (split Mythos vs others) ---
    normal = data[data["model"] != "Claude Mythos Preview"]
    mythos = data[data["model"] == "Claude Mythos Preview"]

    for org, group in normal.groupby("organization"):
        fig.add_trace(go.Scatter(
            x=group["year"],
            y=group["score"],
            mode="markers+lines",
            marker=dict(size=9, opacity=0.9),
            line=dict(width=1, dash="dot"),
            name=org,
            text=group["model"],
            hovertemplate="<b>%{text}</b><br>Year: %{x:.2f}<br>Score: %{y:.1f}%<extra></extra>",
        ))

    if not mythos.empty:
        fig.add_trace(go.Scatter(
            x=mythos["year"],
            y=mythos["score"],
            mode="markers",
            marker=dict(
                size=16,
                color=THEME["mythos"],
                symbol="star",
                line=dict(color="white", width=1.5),
            ),
            name="Claude Mythos Preview ⭐",
            text=mythos["model"],
            hovertemplate="<b>%{text}</b><br>Year: %{x:.2f}<br>Score: %{y:.1f}%<extra></extra>",
        ))

    benchmark_labels = {
        "swe_bench": "SWE-bench Verified — Autonomous Bug Resolution (%)",
        "humaneval": "HumanEval — Code Generation pass@1 (%)",
        "mmlu": "MMLU — Multitask Knowledge Accuracy (%)",
    }

    fig.update_yaxes(title="Score (%)", range=[0, 105])
    fig.update_xaxes(title="Year")

    _apply_theme(
        fig,
        title=benchmark_labels.get(benchmark, benchmark),
        subtitle=(
            "Calibrated sigmoid fit with 95% Monte Carlo uncertainty band | "
            "Sources: Papers with Code, model cards"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Unemployment Scenario Fan Chart
# ---------------------------------------------------------------------------

def plot_unemployment_scenarios(df: pd.DataFrame) -> go.Figure:
    """
    Fan chart showing unemployment projections across all scenarios.
    Median line + P25-P75 band + P5-P95 outer band per scenario.
    """
    fig = go.Figure()

    scenarios = df["scenario_key"].unique()

    for key in scenarios:
        s = df[df["scenario_key"] == key]
        color = s["color"].iloc[0]
        name = s["scenario_name"].iloc[0]

        # Outer band (5–95)
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=pd.concat([s["year"], s["year"].iloc[::-1]]),
            y=pd.concat([s["p95_pct"], s["p5_pct"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # IQR band (25–75)
        fig.add_trace(go.Scatter(
            x=pd.concat([s["year"], s["year"].iloc[::-1]]),
            y=pd.concat([s["p75_pct"], s["p25_pct"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=s["year"],
            y=s["median_pct"],
            mode="lines",
            line=dict(color=color, width=2.5),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Median: %{{y:.1f}}%<extra></extra>",
        ))

    # Milestone markers
    for year, label in [(2030, "~AGI Threshold"), (2026, "Mythos Preview")]:
        fig.add_vline(
            x=year, line_dash="dot", line_color=THEME["text_muted"], line_width=1,
            annotation_text=label,
            annotation_font_color=THEME["text_muted"],
            annotation_position="top right",
        )

    fig.update_yaxes(title="Global Unemployment Rate (%)", range=[0, 75])
    fig.update_xaxes(title="Year", range=[2025, 2050])

    _apply_theme(
        fig,
        title="Global Unemployment Projections — AI Automation Scenarios",
        subtitle=(
            "Monte Carlo simulation (n=5,000) | Bands: P5–P95 outer, P25–P75 inner | "
            "Sector risk sources: McKinsey, WEF, OECD"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Sector Risk Heatmap
# ---------------------------------------------------------------------------

def plot_sector_risk_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing expected job displacement by sector at peak automation.
    """
    df = sector_df.sort_values("displaced_jobs_M")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["sector"],
        x=df["displaced_jobs_M"],
        orientation="h",
        marker=dict(
            color=df["displacement_pct"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Displacement %", tickfont=dict(color=THEME["text"])),
            showscale=True,
        ),
        text=df["displaced_jobs_M"].apply(lambda x: f"{x:.0f}M"),
        textposition="outside",
        textfont=dict(color=THEME["text"]),
        customdata=df[["employment_2025_M", "automation_risk_pct", "displacement_pct"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Displaced: %{x:.1f}M jobs<br>"
            "2025 employment: %{customdata[0]:.0f}M<br>"
            "Automation risk: %{customdata[1]:.0f}%<br>"
            "Displacement rate: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    ))

    fig.update_xaxes(title="Projected Displaced Jobs (Millions) by 2040")
    fig.update_yaxes(title="")

    _apply_theme(
        fig,
        title="Sector-Level Job Displacement at Peak Automation (~2040)",
        subtitle=(
            "Base scenario | Color intensity = % of sector workforce displaced | "
            "Source: McKinsey, WEF, OECD"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Monte Carlo Fan — single scenario deep dive
# ---------------------------------------------------------------------------

def plot_monte_carlo_fan(
    results: dict,
    scenario_key: str = "base",
    years: np.ndarray | None = None,
) -> go.Figure:
    """
    Shows the full distribution of Monte Carlo samples for one scenario,
    including individual sample paths (faded) behind the statistics.
    """
    res = results[scenario_key]
    y = years if years is not None else np.arange(2025, 2051)
    samples = res["samples"]
    color = res["scenario"].color

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()

    # Sample paths (first 200 for visual clarity)
    for i in range(min(200, samples.shape[0])):
        fig.add_trace(go.Scatter(
            x=y, y=samples[i] * 100,
            mode="lines",
            line=dict(color=f"rgba({r},{g},{b},0.03)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    for lo, hi, alpha in [(5, 95, 0.15), (25, 75, 0.30)]:
        fig.add_trace(go.Scatter(
            x=np.concatenate([y, y[::-1]]),
            y=np.concatenate([res[f"p{hi}"] * 100, (res[f"p{lo}"] * 100)[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},{alpha})",
            line=dict(width=0),
            name=f"P{lo}–P{hi}",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=y, y=res["median"] * 100,
        mode="lines",
        line=dict(color=color, width=3),
        name="Median",
    ))

    fig.update_yaxes(title="Unemployment Rate (%)", range=[0, 80])
    fig.update_xaxes(title="Year")

    _apply_theme(
        fig,
        title=f"Monte Carlo Distribution — {res['scenario'].name}",
        subtitle=(
            f"n={samples.shape[0]:,} simulations | Faint lines = individual paths | "
            f"{res['scenario'].description[:80]}..."
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Metaculus AGI Forecast Timeline
# ---------------------------------------------------------------------------

def plot_metaculus_agi_forecast(forecast_df: pd.DataFrame) -> go.Figure:
    """
    Visualize Metaculus community predictions for AGI-related questions.
    """
    if forecast_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Metaculus data available (API may be unreachable)",
            showarrow=False, font=dict(size=14, color=THEME["text_muted"])
        )
        _apply_theme(fig, "Metaculus AGI Forecast Summary")
        return fig

    df = forecast_df.dropna(subset=["community_median"]).copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["title"].apply(lambda t: t[:50] + "..." if len(t) > 50 else t),
        y=df["community_median"] * 100,
        error_y=dict(
            type="data",
            symmetric=False,
            array=((df["community_q3"] - df["community_median"]) * 100).fillna(0),
            arrayminus=((df["community_median"] - df["community_q1"]) * 100).fillna(0),
            color=THEME["text_muted"],
        ),
        marker=dict(color=THEME["accent"], opacity=0.85),
        text=(df["community_median"] * 100).apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color=THEME["text"]),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Community median: %{y:.1f}%<br>"
            "N predictions: %{customdata}"
            "<extra></extra>"
        ),
        customdata=df["num_predictions"].fillna("N/A"),
    ))

    fig.update_xaxes(title="", tickangle=-25)
    fig.update_yaxes(title="Community Probability (%)", range=[0, 110])

    _apply_theme(
        fig,
        title="Metaculus Community Forecasts — AGI Probability Questions",
        subtitle=(
            "Real-time crowd-aggregated predictions | Error bars = IQR (Q1–Q3) | "
            "Source: metaculus.com"
        ),
    )
    return fig
