"""
Plotly figures — light professional theme.

Each function returns a Plotly Figure that renders identically in
notebooks, Streamlit, and saved HTML/PNG/SVG. Colours and typography
are centralised in THEME for one-line restyling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme constants — light palette
# ---------------------------------------------------------------------------

THEME = {
    "bg": "#ffffff",
    "paper": "#ffffff",
    "grid": "#e5e7eb",
    "axis": "#9ca3af",
    "text": "#111827",
    "text_muted": "#6b7280",
    "accent": "#1f6feb",
    "accent2": "#f97316",
    "accent3": "#16a34a",
    "mythos": "#7c3aed",
}

FONT = dict(family="Inter, -apple-system, sans-serif", color=THEME["text"], size=13)

BASE_LAYOUT = dict(
    paper_bgcolor=THEME["paper"],
    plot_bgcolor=THEME["bg"],
    font=FONT,
    xaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["axis"],
        tickfont=dict(color=THEME["text_muted"]),
    ),
    yaxis=dict(
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        linecolor=THEME["axis"],
        tickfont=dict(color=THEME["text_muted"]),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=THEME["grid"],
        borderwidth=1,
        font=dict(color=THEME["text"]),
    ),
    margin=dict(l=60, r=30, t=70, b=60),
    hoverlabel=dict(bgcolor="white", font_color=THEME["text"], bordercolor=THEME["grid"]),
)


def _apply_theme(fig: go.Figure, title: str = "", subtitle: str = "") -> go.Figure:
    if subtitle:
        full_title = (
            f"<b>{title}</b>"
            f"<br><span style='font-size:12px;color:{THEME['text_muted']}'>{subtitle}</span>"
        )
    else:
        full_title = f"<b>{title}</b>"
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text=full_title, font=dict(size=18, color=THEME["text"]), x=0.0, xanchor="left"),
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
    data = df[df["benchmark"] == benchmark].sort_values("year").reset_index(drop=True)

    x_min = float(data["year"].min()) - 0.4
    x_max = float(data["year"].max()) + 0.7
    proj_years = projection_years if projection_years is not None else np.linspace(
        x_min, x_max, 400
    )
    proj_years = proj_years[(proj_years >= x_min) & (proj_years <= x_max)]

    fig = go.Figure()

    if fit_results and benchmark in fit_results:
        fit = fit_results[benchmark]
        mean, lower, upper = fit.predict_with_uncertainty(proj_years)

        fig.add_trace(go.Scatter(
            x=np.concatenate([proj_years, proj_years[::-1]]),
            y=np.concatenate([upper * 100, (lower * 100)[::-1]]),
            fill="toself",
            fillcolor="rgba(31,111,235,0.10)",
            line=dict(width=0),
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=proj_years,
            y=mean * 100,
            mode="lines",
            line=dict(color=THEME["accent"], width=3),
            name="Fitted sigmoid",
        ))

    normal = data[data["model"] != "Claude Mythos Preview"]
    mythos = data[data["model"] == "Claude Mythos Preview"]

    if not normal.empty:
        fig.add_trace(go.Scatter(
            x=normal["year"],
            y=normal["score"],
            mode="markers",
            marker=dict(
                size=10,
                color="#1f2328",
                opacity=0.85,
                line=dict(color="white", width=1.5),
            ),
            name="Published model scores",
            text=normal["model"] + " · " + normal["organization"],
            hovertemplate="<b>%{text}</b><br>%{x:.2f} → %{y:.1f}%<extra></extra>",
        ))

    if not mythos.empty:
        fig.add_trace(go.Scatter(
            x=mythos["year"],
            y=mythos["score"],
            mode="markers",
            marker=dict(
                size=20,
                color=THEME["mythos"],
                symbol="star",
                line=dict(color="white", width=2),
            ),
            name="Claude Mythos Preview (Apr 2026)",
            text=mythos["model"],
            hovertemplate="<b>%{text}</b><br>%{x:.2f} → %{y:.1f}%<extra></extra>",
        ))

    if not data.empty and not mythos.empty:
        first = data.iloc[0]
        last = mythos.iloc[0]
        delta_pp = float(last["score"]) - float(first["score"])
        delta_yr = float(last["year"]) - float(first["year"])

        fig.add_annotation(
            x=float(first["year"]), y=float(first["score"]),
            text=f"<b>{int(first['year'])}</b>: only {first['score']:.1f}% of bugs",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=THEME["text_muted"],
            ax=70, ay=-40,
            font=dict(size=12, color=THEME["text"]),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=THEME["grid"], borderwidth=1, borderpad=5,
        )

        fig.add_annotation(
            x=float(last["year"]), y=float(last["score"]),
            text=(
                f"<b>Apr {int(last['year'])}</b>: {last['score']:.1f}%<br>"
                f"<span style='color:{THEME['mythos']}'>"
                f"+{delta_pp:.0f}pp in {delta_yr:.0f} years</span>"
            ),
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=THEME["mythos"],
            ax=-90, ay=40,
            font=dict(size=13, color=THEME["text"]),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=THEME["mythos"], borderwidth=1.5, borderpad=6,
        )

    benchmark_labels = {
        "swe_bench": "AI's autonomous bug-fixing rate, year by year",
        "humaneval": "AI's code-generation pass rate, year by year",
        "mmlu": "AI's multitask knowledge accuracy, year by year",
    }

    fig.update_yaxes(title="Bugs solved autonomously (%)", range=[0, 105])
    fig.update_xaxes(title="", range=[x_min, x_max])

    _apply_theme(
        fig,
        title=benchmark_labels.get(benchmark, benchmark),
        subtitle=(
            "Sigmoid fitted to every published score · Shaded band = 95% CI · "
            "Sources: Papers with Code, model cards"
        ),
    )
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
    ))
    return fig


# ---------------------------------------------------------------------------
# 2. Unemployment Scenario Fan Chart
# ---------------------------------------------------------------------------

def plot_unemployment_scenarios(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    scenarios = df["scenario_key"].unique()

    for key in scenarios:
        s = df[df["scenario_key"] == key]
        color = s["color"].iloc[0]
        name = s["scenario_name"].iloc[0]

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

        fig.add_trace(go.Scatter(
            x=pd.concat([s["year"], s["year"].iloc[::-1]]),
            y=pd.concat([s["p75_pct"], s["p25_pct"].iloc[::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.20)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=s["year"],
            y=s["median_pct"],
            mode="lines",
            line=dict(color=color, width=2.5),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Median: %{{y:.1f}}%<extra></extra>",
        ))

    for year, label in [(2030, "~AGI threshold"), (2026, "Mythos Preview")]:
        fig.add_vline(
            x=year, line_dash="dot", line_color=THEME["text_muted"], line_width=1,
            annotation_text=label,
            annotation_font_color=THEME["text_muted"],
            annotation_position="top right",
        )

    fig.update_yaxes(title="Global unemployment rate (%)", range=[0, 75])
    fig.update_xaxes(title="Year", range=[2025, 2050])

    _apply_theme(
        fig,
        title="Global unemployment projections — AI automation scenarios",
        subtitle=(
            "Monte Carlo (n=5,000) · Bands: P5–P95 outer, P25–P75 inner · "
            "Sector risk: McKinsey, WEF, OECD"
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Sector Risk Heatmap
# ---------------------------------------------------------------------------

def plot_sector_risk_heatmap(sector_df: pd.DataFrame) -> go.Figure:
    df = sector_df.sort_values("displacement_pct")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["sector"],
        x=df["displacement_pct"],
        orientation="h",
        marker=dict(
            color=df["displacement_pct"],
            colorscale="RdYlGn_r",
            cmin=0,
            cmax=60,
            showscale=False,
            line=dict(color="white", width=1),
        ),
        text=df["displacement_pct"].apply(lambda v: f"<b>{v:.0f}%</b> of jobs lost"),
        textposition="outside",
        textfont=dict(color=THEME["text"], size=13),
        customdata=df[["employment_2025_M", "automation_risk_pct", "displaced_jobs_M"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Workforce displaced: %{x:.1f}%<br>"
            "Jobs displaced: %{customdata[2]:.1f}M<br>"
            "2025 employment: %{customdata[0]:.0f}M<br>"
            "Automation risk: %{customdata[1]:.0f}%"
            "<extra></extra>"
        ),
    ))

    fig.update_xaxes(title="Share of sector workforce displaced by 2040 (%)", range=[0, 75])
    fig.update_yaxes(title="")

    _apply_theme(
        fig,
        title="Which sectors lose the most jobs to AI automation",
        subtitle=(
            "Base scenario, peak automation ~2040 · Bar = % of workers in that sector · "
            "Label also shows absolute jobs displaced · Source: McKinsey, WEF, OECD"
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
    res = results[scenario_key]
    y = years if years is not None else np.arange(2025, 2051)
    samples = res["samples"]
    color = res["scenario"].color

    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    fig = go.Figure()

    for i in range(min(200, samples.shape[0])):
        fig.add_trace(go.Scatter(
            x=y, y=samples[i] * 100,
            mode="lines",
            line=dict(color=f"rgba({r},{g},{b},0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

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

    fig.update_yaxes(title="Unemployment rate (%)", range=[0, 80])
    fig.update_xaxes(title="Year")

    _apply_theme(
        fig,
        title=f"Monte Carlo distribution — {res['scenario'].name}",
        subtitle=f"n={samples.shape[0]:,} simulations · Faint lines = individual paths",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Metaculus AGI Forecast Timeline
# ---------------------------------------------------------------------------

def plot_metaculus_agi_forecast(forecast_df: pd.DataFrame) -> go.Figure:
    if forecast_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No Metaculus data available (API may be unreachable)",
            showarrow=False, font=dict(size=14, color=THEME["text_muted"])
        )
        _apply_theme(fig, "Metaculus AGI forecast summary")
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
    fig.update_yaxes(title="Community probability (%)", range=[0, 110])

    _apply_theme(
        fig,
        title="Metaculus community forecasts — AGI probability questions",
        subtitle=(
            "Real-time crowd-aggregated predictions · Error bars = IQR (Q1–Q3) · "
            "Source: metaculus.com"
        ),
    )
    return fig
