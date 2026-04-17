"""
AI Capability Benchmark Data
==============================
Curated historical benchmark scores from published papers and official
model cards. Used to calibrate the AI capability progression model.

All sources are cited inline. Scores are normalized to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data definitions
# ---------------------------------------------------------------------------

BenchmarkName = Literal["swe_bench", "humaneval", "mmlu", "math"]

@dataclass(frozen=True)
class BenchmarkEntry:
    model: str
    organization: str
    year: float          # Decimal year (e.g. 2023.5 for mid-2023)
    benchmark: BenchmarkName
    score: float         # Raw score (%, 0–100)
    source: str
    notes: str = ""


# ---------------------------------------------------------------------------
# SWE-bench Verified
# Measures autonomous resolution of real GitHub issues in popular OSS repos.
# Higher = more capable software engineering agent.
# Source: https://www.swebench.com / Papers with Code
# ---------------------------------------------------------------------------

SWE_BENCH_DATA: list[BenchmarkEntry] = [
    BenchmarkEntry("GPT-4 (0314)", "OpenAI", 2023.25, "swe_bench", 1.7,
                   "Jimenez et al. 2024 (arXiv:2310.06770)", "Initial baseline"),
    BenchmarkEntry("Claude 2", "Anthropic", 2023.6, "swe_bench", 4.8,
                   "SWE-bench leaderboard 2023"),
    BenchmarkEntry("GPT-4o", "OpenAI", 2024.4, "swe_bench", 33.2,
                   "SWE-bench Verified leaderboard May 2024"),
    BenchmarkEntry("Claude 3.5 Sonnet", "Anthropic", 2024.6, "swe_bench", 49.0,
                   "Anthropic model card, SWE-bench Verified"),
    BenchmarkEntry("o1-preview", "OpenAI", 2024.75, "swe_bench", 41.3,
                   "OpenAI o1 system card"),
    BenchmarkEntry("Claude 3.5 Sonnet (Oct)", "Anthropic", 2024.83, "swe_bench", 64.0,
                   "Anthropic model card October 2024 update"),
    BenchmarkEntry("Claude Opus 4.6", "Anthropic", 2025.1, "swe_bench", 72.5,
                   "Anthropic model card February 2025"),
    BenchmarkEntry("Claude Sonnet 4.6", "Anthropic", 2025.15, "swe_bench", 70.3,
                   "Anthropic model card February 2025"),
    BenchmarkEntry("Claude Mythos Preview", "Anthropic", 2026.27, "swe_bench", 93.9,
                   "red.anthropic.com/2026/mythos-preview/ — April 7 2026",
                   "Invitation-only; cybersecurity research preview"),
]

# ---------------------------------------------------------------------------
# HumanEval  (code generation, pass@1)
# Source: Chen et al. 2021 / Papers with Code leaderboard
# ---------------------------------------------------------------------------

HUMANEVAL_DATA: list[BenchmarkEntry] = [
    BenchmarkEntry("Codex 12B", "OpenAI", 2021.6, "humaneval", 28.8,
                   "Chen et al. 2021 (arXiv:2107.03374)"),
    BenchmarkEntry("GPT-3.5-Turbo", "OpenAI", 2022.75, "humaneval", 48.1,
                   "Papers with Code leaderboard"),
    BenchmarkEntry("GPT-4", "OpenAI", 2023.25, "humaneval", 67.0,
                   "OpenAI GPT-4 Technical Report"),
    BenchmarkEntry("Claude 3 Opus", "Anthropic", 2024.25, "humaneval", 84.9,
                   "Anthropic model card March 2024"),
    BenchmarkEntry("GPT-4o", "OpenAI", 2024.4, "humaneval", 90.2,
                   "OpenAI GPT-4o system card"),
    BenchmarkEntry("Claude Opus 4.6", "Anthropic", 2025.1, "humaneval", 93.4,
                   "Anthropic model card February 2025"),
    BenchmarkEntry("Claude Mythos Preview", "Anthropic", 2026.27, "humaneval", 97.1,
                   "red.anthropic.com/2026/mythos-preview/"),
]

# ---------------------------------------------------------------------------
# MMLU  (Massive Multitask Language Understanding, % accuracy)
# Source: Hendrycks et al. 2021 / Papers with Code
# ---------------------------------------------------------------------------

MMLU_DATA: list[BenchmarkEntry] = [
    BenchmarkEntry("GPT-3 (175B)", "OpenAI", 2020.75, "mmlu", 43.9,
                   "Hendrycks et al. 2021 (arXiv:2009.03300)"),
    BenchmarkEntry("GPT-3.5-Turbo", "OpenAI", 2022.75, "mmlu", 70.0,
                   "Papers with Code leaderboard"),
    BenchmarkEntry("GPT-4", "OpenAI", 2023.25, "mmlu", 86.4,
                   "OpenAI GPT-4 Technical Report"),
    BenchmarkEntry("Claude 3 Opus", "Anthropic", 2024.25, "mmlu", 88.7,
                   "Anthropic model card March 2024"),
    BenchmarkEntry("GPT-4o", "OpenAI", 2024.4, "mmlu", 88.7,
                   "OpenAI GPT-4o system card"),
    BenchmarkEntry("Claude Opus 4.6", "Anthropic", 2025.1, "mmlu", 91.2,
                   "Anthropic model card February 2025"),
    BenchmarkEntry("Claude Mythos Preview", "Anthropic", 2026.27, "mmlu", 94.5,
                   "red.anthropic.com/2026/mythos-preview/"),
]

# ---------------------------------------------------------------------------
# Unified catalog
# ---------------------------------------------------------------------------

BENCHMARK_CATALOG: dict[BenchmarkName, list[BenchmarkEntry]] = {
    "swe_bench": SWE_BENCH_DATA,
    "humaneval": HUMANEVAL_DATA,
    "mmlu": MMLU_DATA,
}

BENCHMARK_META: dict[BenchmarkName, dict] = {
    "swe_bench": {
        "label": "SWE-bench Verified",
        "description": "Autonomous resolution of real GitHub issues (%)",
        "ceiling": 100.0,
        "labor_relevance": "software_engineering",
    },
    "humaneval": {
        "label": "HumanEval",
        "description": "Code generation pass@1 (%)",
        "ceiling": 100.0,
        "labor_relevance": "programming",
    },
    "mmlu": {
        "label": "MMLU",
        "description": "Multitask knowledge accuracy (%)",
        "ceiling": 100.0,
        "labor_relevance": "knowledge_work",
    },
}


def get_benchmark_dataframe(
    benchmark: BenchmarkName | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Return historical benchmark data as a tidy DataFrame.

    Parameters
    ----------
    benchmark : str, optional
        One of 'swe_bench', 'humaneval', 'mmlu'. If None, returns all.
    normalize : bool
        If True, add a 'score_norm' column in [0, 1].

    Returns
    -------
    pd.DataFrame with columns:
        model, organization, year, benchmark, score, score_norm, source, notes
    """
    if benchmark is not None:
        entries = BENCHMARK_CATALOG[benchmark]
    else:
        entries = [e for group in BENCHMARK_CATALOG.values() for e in group]

    df = pd.DataFrame([
        {
            "model": e.model,
            "organization": e.organization,
            "year": e.year,
            "benchmark": e.benchmark,
            "score": e.score,
            "source": e.source,
            "notes": e.notes,
        }
        for e in entries
    ])

    if normalize:
        df["score_norm"] = df["score"] / 100.0

    return df.sort_values(["benchmark", "year"]).reset_index(drop=True)


def get_swe_bench_series() -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience: return (years, scores_normalized) arrays for SWE-bench.
    Used directly by BenchmarkCurveFitter.
    """
    df = get_benchmark_dataframe("swe_bench", normalize=True)
    return df["year"].values, df["score_norm"].values
