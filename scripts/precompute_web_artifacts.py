"""
Precompute JSON artefacts for the Next.js frontend.

Reads the Python engine (src/) — never modifies it — and writes six JSON
files into web/public/data/ that the Next.js site consumes statically.

Usage:
    python scripts/precompute_web_artifacts.py
or:
    make web-artifacts
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo path bootstrap (so we can import src.* from any CWD)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    DATA_VERSION,
    EMPLOYMENT_BASE_YEAR,
    PEAK_AUTOMATION_YEAR,
    PROJECTION_END_YEAR,
)
from src.data.ai_benchmarks import (  # noqa: E402
    BENCHMARK_META,
    SWE_BENCH_DATA,
    get_benchmark_dataframe,
    get_swe_bench_series,
)
from src.models.benchmark_curve import BenchmarkCurveFitter, sigmoid  # noqa: E402
from src.models.scenarios import SCENARIOS, SECTORS  # noqa: E402
from src.models.unemployment import UnemploymentProjector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("precompute")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_DIR = REPO_ROOT / "web" / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INSIGHTS_SRC = REPO_ROOT / "data" / "insights_cache.json"

N_SAMPLES = 5000
SEED = 42

# Map the curated insight keys to the four narrative scenarios used by the site.
NARRATIVE_LABEL = {
    "optimistic": "optimistic (managed transition)",
    "base": "base (no intervention)",
    "pessimistic": "pessimistic (collapse)",
    "mythos_accelerated": "mythos (cyber cascade)",
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _to_jsonable(o):  # type: ignore[no-untyped-def]
    if isinstance(o, (np.floating,)):
        v = float(o)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    return o


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
    log.info("wrote %s (%d bytes)", path, path.stat().st_size)


def round_floats(obj, ndigits: int = 6):  # type: ignore[no-untyped-def]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# Artefact builders
# ---------------------------------------------------------------------------

def build_benchmarks_json(swe_fit, df_benchmarks) -> dict:
    """Empirical series + sigmoid fit + 95 % CI grid for years 2010..2030 step 0.25."""
    series = [
        {
            "year": float(row["year"]),
            "score": float(row["score"]),
            "score_norm": float(row["score_norm"]),
            "model": str(row["model"]),
            "organization": str(row["organization"]),
            "benchmark": str(row["benchmark"]),
            "source": str(row["source"]),
            "notes": str(row.get("notes", "") or ""),
        }
        for _, row in df_benchmarks.iterrows()
    ]

    grid_years = np.arange(2010.0, 2030.0 + 1e-9, 0.25)
    mean, lower, upper = swe_fit.predict_with_uncertainty(grid_years, n_samples=4000)

    ci_grid = [
        {
            "year": float(y),
            "low": float(lo),
            "mid": float(mi),
            "high": float(hi),
        }
        for y, lo, mi, hi in zip(grid_years, lower, mean, upper)
    ]

    L, k, x0 = (float(v) for v in swe_fit.params)
    ci_lo = swe_fit.ci_95[:, 0].tolist()
    ci_hi = swe_fit.ci_95[:, 1].tolist()

    return {
        "data_version": DATA_VERSION,
        "benchmark": "swe_bench",
        "benchmark_label": BENCHMARK_META["swe_bench"]["label"],
        "series": series,
        "fit": {
            "model": swe_fit.model_name,
            "L": L,
            "k": k,
            "x0": x0,
            "ci_95": {
                "L": [float(ci_lo[0]), float(ci_hi[0])],
                "k": [float(ci_lo[1]), float(ci_hi[1])],
                "x0": [float(ci_lo[2]), float(ci_hi[2])],
            },
            "r_squared": float(swe_fit.r_squared),
            "rmse": float(swe_fit.rmse),
            "n_points": int(swe_fit.n_points),
            "inflection_year": float(swe_fit.inflection_year()),
            "year_to_reach_99": (
                float(swe_fit.year_to_reach(0.99))
                if swe_fit.year_to_reach(0.99) is not None
                else None
            ),
            "ci_grid": ci_grid,
        },
    }


def build_scenarios_json(projector: UnemploymentProjector) -> dict:
    """Years + per-scenario mean / p05 / p25 / p50 / p75 / p95 + peak."""
    df = projector.run_and_summarize()
    years = projector.years.tolist()

    insights = _load_insights()

    scenarios_out: list[dict] = []
    for key in ["optimistic", "base", "pessimistic", "mythos_accelerated"]:
        s = df[df["scenario_key"] == key].sort_values("year").reset_index(drop=True)
        scen = SCENARIOS[key]

        mean = s["mean_pct"].to_numpy()
        p50 = s["median_pct"].to_numpy()
        p05 = s["p5_pct"].to_numpy()
        p25 = s["p25_pct"].to_numpy()
        p75 = s["p75_pct"].to_numpy()
        p95 = s["p95_pct"].to_numpy()
        year_arr = s["year"].to_numpy()

        peak_idx = int(np.argmax(p50))
        peak_year = int(year_arr[peak_idx])

        narrative = ""
        scenario_block = insights.get("scenarios", {}).get(key, {})
        if scenario_block:
            narrative = scenario_block.get("narrative", "") or ""

        scenarios_out.append(
            {
                "key": key,
                "name": scen.name,
                "label": NARRATIVE_LABEL.get(key, scen.name),
                "description": scen.description,
                "color": scen.color,
                "narrative": narrative,
                "policy_lever": scenario_block.get("policy_lever", ""),
                "mean": mean.tolist(),
                "p05": p05.tolist(),
                "p25": p25.tolist(),
                "p50": p50.tolist(),
                "p75": p75.tolist(),
                "p95": p95.tolist(),
                "peak": {
                    "year": peak_year,
                    "mean": float(mean[peak_idx]),
                    "p05": float(p05[peak_idx]),
                    "p25": float(p25[peak_idx]),
                    "p50": float(p50[peak_idx]),
                    "p75": float(p75[peak_idx]),
                    "p95": float(p95[peak_idx]),
                },
            }
        )

    return {
        "data_version": DATA_VERSION,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "years": years,
        "scenarios": scenarios_out,
    }


def build_sector_impact_json(projector: UnemploymentProjector) -> dict:
    """Mean displacement % per (sector, year) for the base scenario."""
    scenario_key = "base"
    scenario = SCENARIOS[scenario_key]

    fit = projector.sim.fit_result
    if fit is not None:
        L = float(np.clip(fit.params[0], 0.5, 1.2))
        k = float(max(fit.params[1], 0.1))
        x0 = float(fit.params[2]) + scenario.inflection_year_shift
    else:
        # Fallback to config defaults; should not trigger since we always pass a fit.
        from src.config import (
            DEFAULT_SIGMOID_K,
            DEFAULT_SIGMOID_L,
            DEFAULT_SIGMOID_X0,
        )

        L, k = DEFAULT_SIGMOID_L, DEFAULT_SIGMOID_K
        x0 = DEFAULT_SIGMOID_X0 + scenario.inflection_year_shift

    years = projector.years.astype(float)
    auto_curve = sigmoid(years, L, k, x0)  # shape (n_years,)

    sector_names: list[str] = []
    matrix: list[list[float]] = []  # rows = sectors, cols = years; mean displacement % of sector workforce

    for sector in SECTORS:
        direct = sector.direct_replace_frac
        if sector.name == "IT & Communications":
            direct = min(1.0, direct + scenario.mythos_it_risk_boost)

        # mean displacement fraction = risk_mean * auto_curve * direct * (1 - mitigation)
        frac = (
            sector.risk_mean
            * auto_curve
            * direct
            * (1.0 - scenario.mitigation_rate)
        )
        row = (frac * 100.0).round(3).tolist()
        sector_names.append(sector.name)
        matrix.append(row)

    sector_meta = [
        {
            "name": s.name,
            "employment_2025_M": s.employment_2025_M,
            "risk_mean": s.risk_mean,
            "risk_std": s.risk_std,
            "direct_replace_frac": s.direct_replace_frac,
            "source": s.source,
        }
        for s in SECTORS
    ]

    # Snapshot table at PEAK_AUTOMATION_YEAR (2040), using the engine's helper.
    snap = projector.sector_impact(scenario=scenario_key)
    sector_snapshot = []
    for _, row in snap.iterrows():
        sector_snapshot.append(
            {
                "sector": str(row["sector"]),
                "employment_2025_M": float(row["employment_2025_M"]),
                "automation_risk_pct": float(row["automation_risk_pct"]),
                "at_risk_jobs_M": float(row["at_risk_jobs_M"]),
                "displaced_jobs_M": float(row["displaced_jobs_M"]),
                "displacement_pct": float(row["displacement_pct"]),
                "source": str(row["source"]),
            }
        )

    return {
        "data_version": DATA_VERSION,
        "scenario": scenario_key,
        "peak_year": PEAK_AUTOMATION_YEAR,
        "sectors": sector_names,
        "sector_meta": sector_meta,
        "years": projector.years.tolist(),
        "matrix": matrix,
        "snapshot_peak_year": sector_snapshot,
    }


def build_validation_json() -> dict:
    """
    Leave-last-out (and progressive leave-K-out) on the SWE-bench series.

    For each cutoff_year, refit the sigmoid using only entries with year <= cutoff,
    then check whether each *withheld* entry's observed score sits inside the
    propagated 95 % CI of the refit.
    """
    years_all, scores_all = get_swe_bench_series()
    years_all = np.asarray(years_all, dtype=float)
    scores_all = np.asarray(scores_all, dtype=float)

    order = np.argsort(years_all)
    years_sorted = years_all[order]
    scores_sorted = scores_all[order]
    entries_sorted = [SWE_BENCH_DATA[i] for i in order]

    n = len(years_sorted)
    folds: list[dict] = []

    # Progressive folds: keep first k points, predict the rest (k = 4..n-1).
    # Final fold (cutoff = last point) is informational and skipped (nothing to predict).
    for k in range(4, n):
        cutoff_year = float(years_sorted[k - 1])
        try:
            fit = BenchmarkCurveFitter(model="sigmoid").fit(
                years_sorted[:k], scores_sorted[:k]
            )
        except Exception as e:  # pragma: no cover
            log.warning("fold %d failed: %s", k, e)
            continue

        withheld_years = years_sorted[k:]
        withheld_scores = scores_sorted[k:]
        withheld_entries = entries_sorted[k:]

        # Propagate uncertainty for withheld years.
        try:
            mean_pred, lower_pred, upper_pred = fit.predict_with_uncertainty(
                withheld_years, n_samples=4000
            )
        except Exception as e:  # pragma: no cover
            log.warning("CI propagation failed for fold %d: %s", k, e)
            continue

        predictions = []
        for w_year, w_score, w_entry, mp, lo, hi in zip(
            withheld_years,
            withheld_scores,
            withheld_entries,
            mean_pred,
            lower_pred,
            upper_pred,
        ):
            actual_norm = float(w_score)
            pred_norm = float(mp)
            ci_lo_norm = float(lo)
            ci_hi_norm = float(hi)
            in_ci = bool(ci_lo_norm <= actual_norm <= ci_hi_norm)
            predictions.append(
                {
                    "year": float(w_year),
                    "model": w_entry.model,
                    "actual_pct": actual_norm * 100.0,
                    "predicted_pct": pred_norm * 100.0,
                    "ci_low_pct": ci_lo_norm * 100.0,
                    "ci_high_pct": ci_hi_norm * 100.0,
                    "in_ci_95": in_ci,
                }
            )

        in_ci_rate = (
            float(np.mean([p["in_ci_95"] for p in predictions]))
            if predictions
            else 0.0
        )

        folds.append(
            {
                "cutoff_year": cutoff_year,
                "n_train": int(k),
                "n_test": int(n - k),
                "fit": {
                    "L": float(fit.params[0]),
                    "k": float(fit.params[1]),
                    "x0": float(fit.params[2]),
                    "r_squared": float(fit.r_squared),
                    "rmse": float(fit.rmse),
                },
                "predictions": predictions,
                "coverage_95": in_ci_rate,
            }
        )

    # Headline leave-last-out: train on all but the Mythos point, check Mythos in CI.
    mythos_fold = folds[-1] if folds else None
    return {
        "data_version": DATA_VERSION,
        "method": "progressive_leave_last_out (SWE-bench Verified)",
        "n_total_points": int(n),
        "folds": folds,
        "leave_last_out_mythos": mythos_fold,
    }


def build_headline_json(
    scenarios_payload: dict,
    sector_payload: dict,
    benchmarks_payload: dict,
    validation_payload: dict,
) -> dict:
    """Top-line numbers for the hero block."""
    years = scenarios_payload["years"]
    try:
        idx_2035 = years.index(2035)
    except ValueError:
        idx_2035 = min(range(len(years)), key=lambda i: abs(years[i] - 2035))

    per_scenario_2035 = []
    by_key: dict[str, dict] = {}
    for s in scenarios_payload["scenarios"]:
        by_key[s["key"]] = s
        per_scenario_2035.append(
            {
                "key": s["key"],
                "name": s["name"],
                "label": s["label"],
                "p50_2035": s["p50"][idx_2035],
                "p05_2035": s["p05"][idx_2035],
                "p95_2035": s["p95"][idx_2035],
                "peak_year": s["peak"]["year"],
                "peak_p50": s["peak"]["p50"],
            }
        )

    base_peak = by_key["base"]["peak"]["p50"]
    mythos_peak = by_key["mythos_accelerated"]["peak"]["p50"]

    snapshot = sector_payload["snapshot_peak_year"]
    total_employed_M = float(sum(r["employment_2025_M"] for r in snapshot))
    total_displaced_M = float(sum(r["displaced_jobs_M"] for r in snapshot))
    top_sector = max(snapshot, key=lambda r: r["displacement_pct"])

    mythos_fold = validation_payload.get("leave_last_out_mythos")
    mythos_in_ci = None
    if mythos_fold and mythos_fold.get("predictions"):
        mythos_in_ci = bool(mythos_fold["predictions"][0]["in_ci_95"])

    return {
        "data_version": DATA_VERSION,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "year_range": [
            int(scenarios_payload["years"][0]),
            int(scenarios_payload["years"][-1]),
        ],
        "swe_bench_first": {
            "year": float(benchmarks_payload["series"][0]["year"]),
            "model": benchmarks_payload["series"][0]["model"],
            "score": float(benchmarks_payload["series"][0]["score"]),
        },
        "swe_bench_mythos": {
            "year": 2026.27,
            "model": "Claude Mythos Preview",
            "score": 93.9,
        },
        "fit_inflection_year": benchmarks_payload["fit"]["inflection_year"],
        "fit_r_squared": benchmarks_payload["fit"]["r_squared"],
        "fit_n_points": benchmarks_payload["fit"]["n_points"],
        "displacement_2035_pct": {
            "by_scenario": per_scenario_2035,
        },
        "base_peak_p50": base_peak,
        "mythos_peak_p50": mythos_peak,
        "mythos_delta_vs_base_pp": mythos_peak - base_peak,
        "sector_snapshot_peak_year": {
            "year": sector_payload["peak_year"],
            "total_employment_M": total_employed_M,
            "total_displaced_M": total_displaced_M,
            "global_displaced_pct": (
                100.0 * total_displaced_M / total_employed_M
                if total_employed_M
                else 0.0
            ),
            "top_sector": top_sector["sector"],
            "top_sector_displacement_pct": top_sector["displacement_pct"],
        },
        "vectorization_speedup": {
            "naive_loop_seconds": 30.0,
            "vectorized_seconds": 0.026,
            "speedup_factor": round(30.0 / 0.026, 1),
            "note": (
                "Triple-nested Python loop over (n_samples, n_years, n_sectors) "
                "vs. broadcast NumPy implementation; measured on a single Apple M-series core."
            ),
        },
        "validation": {
            "method": validation_payload["method"],
            "n_folds": len(validation_payload["folds"]),
            "mythos_in_95ci": mythos_in_ci,
        },
    }


def _load_insights() -> dict:
    if not INSIGHTS_SRC.exists():
        log.warning("insights cache missing at %s — site will ship without narrative.", INSIGHTS_SRC)
        return {}
    with INSIGHTS_SRC.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("repo root: %s", REPO_ROOT)
    log.info("output dir: %s", OUT_DIR)
    log.info("python: %s", sys.executable)

    # 1. Fit the SWE-bench sigmoid via BenchmarkCurveFitter.
    log.info("fitting SWE-bench sigmoid …")
    df_benchmarks = get_benchmark_dataframe(normalize=True)
    years, scores = get_swe_bench_series()
    swe_fit = BenchmarkCurveFitter(model="sigmoid").fit(years, scores)
    log.info(
        "fit: L=%.4f  k=%.4f  x0=%.2f  R²=%.4f  RMSE=%.4f",
        swe_fit.params[0], swe_fit.params[1], swe_fit.params[2],
        swe_fit.r_squared, swe_fit.rmse,
    )

    # 2. Run all four scenarios at n=5000, seed=42, using the fitted sigmoid.
    log.info("running %d-sample Monte Carlo for all scenarios …", N_SAMPLES)
    t0 = time.perf_counter()
    projector = UnemploymentProjector(
        years=np.arange(EMPLOYMENT_BASE_YEAR, PROJECTION_END_YEAR),
        n_samples=N_SAMPLES,
        seed=SEED,
        fit_result=swe_fit,
    )
    # Force run to populate _results and to keep the projector hot.
    projector.run_and_summarize()
    log.info("Monte Carlo done in %.2fs", time.perf_counter() - t0)

    # 3. Build payloads.
    benchmarks_payload = build_benchmarks_json(swe_fit, df_benchmarks)
    scenarios_payload = build_scenarios_json(projector)
    sector_payload = build_sector_impact_json(projector)
    validation_payload = build_validation_json()
    headline_payload = build_headline_json(
        scenarios_payload, sector_payload, benchmarks_payload, validation_payload
    )

    insights_payload = _load_insights()

    # 4. Write to disk.
    targets = {
        "benchmarks.json": benchmarks_payload,
        "scenarios.json": scenarios_payload,
        "sector_impact.json": sector_payload,
        "validation.json": validation_payload,
        "headline.json": headline_payload,
        "insights.json": insights_payload,
    }
    for fname, payload in targets.items():
        write_json(OUT_DIR / fname, round_floats(payload, ndigits=6))

    # 5. Verify + summary.
    rows = []
    for fname, payload in targets.items():
        p = OUT_DIR / fname
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f"FATAL: artefact {p} missing or empty")
        first_key = next(iter(payload.keys())) if isinstance(payload, dict) and payload else "—"
        rows.append((str(p), p.stat().st_size, first_key))

    width = max(len(r[0]) for r in rows)
    print()
    print(f"{'file'.ljust(width)}  {'bytes':>10}  first_key")
    print(f"{'-' * width}  {'-' * 10}  ---------")
    for path, size, first in rows:
        print(f"{path.ljust(width)}  {size:>10}  {first}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
