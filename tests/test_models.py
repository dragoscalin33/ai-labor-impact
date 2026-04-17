"""
Unit tests for core modeling modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from src.data.ai_benchmarks import get_benchmark_dataframe, get_swe_bench_series
from src.models.benchmark_curve import BenchmarkCurveFitter, sigmoid
from src.models.scenarios import SECTORS, ScenarioSimulator
from src.models.unemployment import UnemploymentProjector


class TestBenchmarkCurveFitter:

    def test_sigmoid_shape(self):
        x = np.array([2020, 2025, 2030, 2035])
        y = sigmoid(x, L=1.0, k=0.5, x0=2030)
        assert y.shape == (4,)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_fit_swe_bench(self):
        years, scores = get_swe_bench_series()
        fitter = BenchmarkCurveFitter(model="sigmoid")
        result = fitter.fit(years, scores)

        assert result.r_squared > 0.90, f"R² too low: {result.r_squared}"
        assert 2020 < result.inflection_year() < 2040
        assert result.rmse < 0.15
        assert result.n_points == len(years)

    def test_fit_result_summary_shape(self):
        years, scores = get_swe_bench_series()
        result = BenchmarkCurveFitter().fit(years, scores)
        summary = result.summary()
        assert len(summary) == 3  # sigmoid has 3 params
        assert "ci_95_lower" in summary.columns

    def test_predict_with_uncertainty_shape(self):
        years, scores = get_swe_bench_series()
        result = BenchmarkCurveFitter().fit(years, scores)
        future = np.arange(2025, 2051)
        mean, lo, hi = result.predict_with_uncertainty(future, n_samples=200)
        assert mean.shape == (len(future),)
        assert np.all(lo <= mean)
        assert np.all(mean <= hi)

    def test_year_to_reach(self):
        years, scores = get_swe_bench_series()
        result = BenchmarkCurveFitter().fit(years, scores)
        y99 = result.year_to_reach(0.99)
        assert y99 is None or (2026 < y99 < 2050)

    def test_fit_all_benchmarks(self):
        df = get_benchmark_dataframe(normalize=True)
        fitter = BenchmarkCurveFitter()
        results = fitter.fit_all_benchmarks(df)
        assert len(results) >= 2
        for bm, res in results.items():
            assert res.r_squared > 0.0


class TestScenarioSimulator:

    def setup_method(self):
        self.sim = ScenarioSimulator(
            years=np.arange(2025, 2036),  # short for speed
            n_samples=50,
            seed=42,
        )

    def test_run_scenario_shape(self):
        result = self.sim.run_scenario("base")
        assert result["samples"].shape == (50, 11)  # 50 samples, 11 years
        assert "mean" in result
        assert "p95" in result

    def test_unemployment_non_negative(self):
        result = self.sim.run_scenario("optimistic")
        assert np.all(result["samples"] >= 0)

    def test_optimistic_lt_pessimistic(self):
        opt = self.sim.run_scenario("optimistic")
        pess = self.sim.run_scenario("pessimistic")
        # Optimistic peak should be lower than pessimistic peak
        assert opt["median"].max() < pess["median"].max()

    def test_to_dataframe_columns(self):
        results = self.sim.run_all()
        df = self.sim.to_dataframe(results)
        expected_cols = {"scenario_key", "year", "median_pct", "p5_pct", "p95_pct"}
        assert expected_cols.issubset(df.columns)

    def test_sector_impact_table(self):
        df = self.sim.sector_impact_table("base")
        assert len(df) == len(SECTORS)
        assert "displaced_jobs_M" in df.columns
        assert df["displaced_jobs_M"].min() >= 0


class TestUnemploymentProjector:

    def setup_method(self):
        self.proj = UnemploymentProjector(
            years=np.arange(2025, 2036),
            n_samples=50,
            seed=42,
        )

    def test_run_and_summarize(self):
        df = self.proj.run_and_summarize(scenarios=["base", "optimistic"])
        assert len(df["scenario_key"].unique()) == 2

    def test_peak_unemployment_ordering(self):
        df = self.proj.run_and_summarize()
        peak = self.proj.peak_unemployment(df)
        vals = peak["peak_unemployment_pct"].values
        # Should be sorted descending
        assert np.all(vals[:-1] >= vals[1:])

    def test_year_crossing(self):
        df = self.proj.run_and_summarize(scenarios=["base"])
        cross = self.proj.year_crossing(1.0, df)  # low threshold, should always cross
        assert len(cross) == 1


class TestBenchmarkData:

    def test_dataframe_shape(self):
        df = get_benchmark_dataframe()
        assert len(df) > 10
        assert "year" in df.columns
        assert "score" in df.columns
        assert "score_norm" in df.columns

    def test_mythos_present(self):
        df = get_benchmark_dataframe("swe_bench")
        mythos = df[df["model"] == "Claude Mythos Preview"]
        assert len(mythos) == 1
        assert float(mythos["score"].iloc[0]) == pytest.approx(93.9, abs=0.1)

    def test_scores_in_valid_range(self):
        df = get_benchmark_dataframe(normalize=True)
        assert df["score"].between(0, 100).all()
        assert df["score_norm"].between(0, 1).all()

    def test_swe_bench_series_monotone(self):
        years, scores = get_swe_bench_series()
        # Generally should be increasing over time
        assert scores[-1] > scores[0], "SWE-bench should improve over time"
