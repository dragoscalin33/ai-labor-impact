"""
Insights Generator
===================
LLM-ready abstraction layer for automatic narrative generation.

Architecture pattern: Strategy / Provider interface.
Swap providers without changing any calling code.

Currently implemented providers:
- TemplateProvider  : rule-based, no API key needed (default)
- GroqProvider      : Llama 3.3 70B via Groq free API
- AnthropicProvider : Claude via Anthropic API

Usage
-----
>>> gen = InsightsGenerator(provider="template")
>>> summary = gen.summarize_scenario_results(df, peak_unemployment_df)
>>> print(summary)

# When you have an API key:
>>> gen = InsightsGenerator(provider="groq", api_key="gsk_...")
>>> summary = gen.summarize_scenario_results(df, peak_unemployment_df)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

INSIGHTS_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "insights_cache.json"
)


def load_insights_cache(path: Path | None = None) -> dict[str, Any] | None:
    """Load the bundled insights cache or return None if it isn't present."""
    p = path or INSIGHTS_CACHE_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load insights cache at %s: %s", p, exc)
        return None


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class BaseInsightsProvider(ABC):
    """All providers implement this interface."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        ...


# ---------------------------------------------------------------------------
# Template provider (no API key)
# ---------------------------------------------------------------------------

class TemplateProvider(BaseInsightsProvider):
    """
    Rule-based insight generator. No API needed.

    Produces professional, data-driven narrative from structured results.
    When ``data/insights_cache.json`` is present, the cached high-quality
    per-scenario commentary is returned verbatim — so the dashboard's
    "Insights" tab is rich even with zero LLM keys configured.
    """

    def __init__(self) -> None:
        self._cache = load_insights_cache()

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        # Template provider returns the pre-formatted prompt directly
        # (In practice, calling code sends pre-built narratives here)
        return prompt

    def scenario_narrative(
        self,
        scenario_name: str,
        peak_pct: float,
        peak_year: int,
        year_10pct: int | None,
        scenario_key: str | None = None,
    ) -> str:
        # 1) Prefer the curated cache if a matching scenario_key is known.
        if scenario_key and self._cache:
            cached = self._cache.get("scenarios", {}).get(scenario_key)
            if cached:
                metrics_block = (
                    f"\n\n*Realised in this run:* peak median **{peak_pct:.1f}%** "
                    f"around {peak_year}"
                    + (f", crossing 10% by {year_10pct}" if year_10pct else "")
                    + "."
                )
                policy_line = (
                    f"\n\n**Policy lever.** {cached.get('policy_lever', '')}"
                    if cached.get("policy_lever") else ""
                )
                return (
                    f"### {cached.get('headline', scenario_name)}\n\n"
                    f"{cached.get('narrative', '')}"
                    f"{policy_line}{metrics_block}"
                )

        # 2) Dynamic fallback — works without the cache file.
        severity = (
            "severe" if peak_pct > 40
            else "significant" if peak_pct > 20
            else "moderate"
        )
        crossing = f"reaching 10% by {year_10pct}" if year_10pct else "remaining below 10%"
        if severity == "severe":
            policy_line = (
                "Policy intervention and large-scale reskilling programs would be "
                "essential to prevent permanent structural unemployment."
            )
        else:
            policy_line = (
                "Proactive investment in human-AI collaboration skills could "
                "substantially mitigate the impact."
            )
        return (
            f"Under the **{scenario_name}**, the model projects a {severity} labor "
            f"market disruption, with unemployment peaking at **{peak_pct:.1f}%** "
            f"around {peak_year}. The transition begins gradually before {crossing}, "
            f"then accelerates as AI capability crosses the autonomous task execution "
            f"threshold. {policy_line}"
        )

    def benchmark_narrative(
        self,
        benchmark: str,
        inflection_year: float,
        year_99pct: float | None,
        r_squared: float,
    ) -> str:
        bm_labels = {
            "swe_bench": "SWE-bench Verified (autonomous software engineering)",
            "humaneval": "HumanEval (code generation)",
            "mmlu": "MMLU (broad knowledge tasks)",
        }
        label = bm_labels.get(benchmark, benchmark)
        year_str = f"~{year_99pct:.0f}" if year_99pct else "beyond the projection window"
        return (
            f"The {label} progression follows a sigmoid curve with inflection point at "
            f"**{inflection_year:.1f}** (model fit R²={r_squared:.3f}). "
            f"At this rate, near-human-expert performance (99%) is projected by {year_str}. "
            f"The Mythos Preview data point (April 2026) sits at the steep phase of the curve, "
            f"suggesting the most rapid capability gains are occurring now."
        )

    def mythos_context(self) -> str:
        return (
            "**Claude Mythos Preview (April 7, 2026)** achieved 93.9% on SWE-bench Verified — "
            "a 21 percentage point jump from Claude Opus 4.6 (72.5%) in approximately 14 months. "
            "This single data point places AI software engineering capability firmly in the "
            "'expert human' range. Mythos autonomously discovered thousands of zero-day "
            "vulnerabilities including a 17-year-old FreeBSD RCE bug, suggesting that "
            "autonomous AI capability is no longer theoretical for high-skilled knowledge work."
        )


# ---------------------------------------------------------------------------
# Groq provider (free tier — Llama 3.3 70B)
# ---------------------------------------------------------------------------

class GroqProvider(BaseInsightsProvider):
    """
    Groq API provider — free tier available at console.groq.com.
    Supports Llama 3.3 70B, Mixtral 8x7B, and others.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("Install groq: pip install groq")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior economist and data scientist specializing in "
                        "labor market impacts of AI. Write concise, rigorous, data-driven "
                        "insights. Use precise language, cite data points, avoid speculation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Anthropic Claude provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseInsightsProvider):
    """
    Anthropic Claude API provider.
    Requires ANTHROPIC_API_KEY environment variable or explicit api_key.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        try:
            import os

            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Set ANTHROPIC_API_KEY environment variable or pass api_key=")
            self.client = anthropic.Anthropic(api_key=key)
            self.model = model
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=(
                "You are a senior economist and ML researcher. "
                "Write concise, precise, data-driven analysis. "
                "Reference specific numbers. No hedging language."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# Main InsightsGenerator
# ---------------------------------------------------------------------------

PROVIDERS = {
    "template": TemplateProvider,
    "groq": GroqProvider,
    "anthropic": AnthropicProvider,
}


class InsightsGenerator:
    """
    Unified interface for generating data-driven narrative insights.

    Parameters
    ----------
    provider : str
        One of 'template' (no API), 'groq' (free), 'anthropic'
    api_key : str, optional
        Required for 'groq' and 'anthropic' providers.
    **kwargs
        Additional arguments passed to the provider (e.g. model=).
    """

    def __init__(
        self,
        provider: str = "template",
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        cls = PROVIDERS.get(provider)
        if cls is None:
            raise ValueError(f"Unknown provider: {provider}. Choose from {list(PROVIDERS)}")

        if provider == "template":
            self._provider = TemplateProvider()
        else:
            self._provider = cls(api_key=api_key, **kwargs)  # type: ignore

        self._template = TemplateProvider()
        self.provider_name = provider
        logger.info(f"InsightsGenerator initialized with provider: {provider}")

    def summarize_scenario_results(
        self,
        df: pd.DataFrame,
        peak_df: pd.DataFrame,
        crossing_df: pd.DataFrame | None = None,
    ) -> dict[str, str]:
        """
        Generate one narrative insight per scenario.

        Returns dict mapping scenario_name → insight text.
        """
        insights = {}
        for _, row in peak_df.iterrows():
            scenario_name = row["scenario_name"]
            peak_pct = row["peak_unemployment_pct"]

            s_df = df[df["scenario_name"] == scenario_name]
            peak_year = int(s_df.loc[s_df["median_pct"].idxmax(), "year"])

            year_10pct = None
            if crossing_df is not None:
                cross = crossing_df[
                    (crossing_df["scenario_name"] == scenario_name) &
                    (crossing_df["threshold_pct"] == 10.0)
                ]
                if not cross.empty and cross["year_crossing"].notna().any():
                    year_10pct = int(cross["year_crossing"].iloc[0])

            if self.provider_name == "template":
                # Try to map scenario_name back to its key (cache uses keys).
                scenario_key = next(
                    (
                        k for k, v in (self._template._cache or {})
                        .get("scenarios", {}).items()
                        if v.get("headline", "").lower() in scenario_name.lower()
                        or scenario_name.lower() in v.get("headline", "").lower()
                    ),
                    None,
                )
                text = self._template.scenario_narrative(
                    scenario_name, peak_pct, peak_year, year_10pct,
                    scenario_key=scenario_key,
                )
            else:
                prompt = (
                    f"Analyze this unemployment projection scenario:\n"
                    f"- Scenario: {scenario_name}\n"
                    f"- Peak unemployment: {peak_pct:.1f}% around {peak_year}\n"
                    f"- 10% threshold crossed: {year_10pct or 'not reached'}\n"
                    f"Write 3 concise sentences: the trend, the main driver, "
                    f"and the key policy implication. Include specific numbers."
                )
                text = self._provider.generate(prompt, max_tokens=200)

            insights[scenario_name] = text

        return insights

    def benchmark_insight(
        self,
        benchmark: str,
        fit_result: Any,
    ) -> str:
        """Generate narrative for a fitted benchmark curve."""
        infl = fit_result.inflection_year()
        year_99 = fit_result.year_to_reach(0.99)
        r2 = fit_result.r_squared

        if self.provider_name == "template":
            return self._template.benchmark_narrative(benchmark, infl, year_99, r2)

        prompt = (
            f"Interpret this AI benchmark trend:\n"
            f"- Benchmark: {benchmark}\n"
            f"- Sigmoid fit R²: {r2:.3f}\n"
            f"- Inflection year (fastest growth): {infl:.1f}\n"
            f"- Year to reach 99% capability: {year_99:.0f if year_99 else 'unknown'}\n"
            f"- The Mythos Preview (April 2026) scored 93.9% on SWE-bench Verified.\n"
            f"Write 2 sentences analyzing what this curve shape means for labor market "
            f"impact timeline. Be specific and use the numbers."
        )
        return self._provider.generate(prompt, max_tokens=150)

    def executive_summary(
        self,
        peak_df: pd.DataFrame,
        benchmark_insights: dict[str, str],
    ) -> str:
        """Generate a concise executive summary of the full analysis."""
        base_peak = peak_df[peak_df["scenario_name"].str.contains("Base", case=False)]
        base_pct = (
            float(base_peak["peak_unemployment_pct"].iloc[0])
            if not base_peak.empty else 0
        )

        if self.provider_name == "template":
            cached = (self._template._cache or {}).get("executive_summary")
            if cached:
                return (
                    f"{cached}\n\n"
                    f"*Realised this run — Base scenario peak: **{base_pct:.1f}%**.*"
                )
            return (
                f"## Executive Summary\n\n"
                f"{self._template.mythos_context()}\n\n"
                f"**Projection Results:** Under the base scenario (no major policy "
                f"intervention), global unemployment from AI automation is projected "
                f"to reach **{base_pct:.1f}%** of the 2025 labor force by peak "
                f"displacement. The optimistic scenario demonstrates that aggressive "
                f"reskilling and new industry development can contain this to under "
                f"15%, while the pessimistic scenario — absent policy action — "
                f"exceeds 55%.\n\n"
                f"**Data quality note:** All projections use real employment data "
                f"(World Bank API), real AI benchmark progression (Papers with Code), "
                f"and real crowd forecasts (Metaculus). Uncertainty is quantified via "
                f"Monte Carlo simulation (n=5,000). Model parameters are fitted to "
                f"data, not assumed."
            )

        prompt = (
            f"Write a 150-word executive summary for a data science portfolio project "
            f"analyzing AI impact on global employment. Key facts:\n"
            f"- Base scenario peak unemployment: {base_pct:.1f}%\n"
            f"- Claude Mythos Preview: 93.9% SWE-bench (April 2026)\n"
            f"- Analysis uses real World Bank data, fitted sigmoid curves, Monte Carlo (n=5000)\n"
            f"- Scenarios range from managed transition to structural collapse\n"
            f"Write for a technical hiring manager. Lead with the most striking finding."
        )
        return self._provider.generate(prompt, max_tokens=200)
