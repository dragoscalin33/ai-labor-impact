export type ScenarioKey =
  | "optimistic"
  | "base"
  | "pessimistic"
  | "mythos_accelerated";

export interface HeadlineScenarioEntry {
  key: ScenarioKey;
  name: string;
  label: string;
  p50_2035: number;
  p05_2035: number;
  p95_2035: number;
  peak_year: number;
  peak_p50: number;
}

export interface HeadlineSectorSnapshot {
  year: number;
  total_employment_M: number;
  total_displaced_M: number;
  global_displaced_pct: number;
  top_sector: string;
  top_sector_displacement_pct: number;
}

export interface HeadlineVectorization {
  naive_loop_seconds: number;
  vectorized_seconds: number;
  speedup_factor: number;
  note: string;
}

export interface HeadlineValidation {
  method: string;
  n_folds: number;
  mythos_in_95ci: boolean;
}

export interface HeadlineSweEvent {
  year: number;
  model: string;
  score: number;
}

export interface HeadlineData {
  data_version: string;
  n_samples: number;
  seed: number;
  year_range: [number, number];
  swe_bench_first: HeadlineSweEvent;
  swe_bench_mythos: HeadlineSweEvent;
  fit_inflection_year: number;
  fit_r_squared: number;
  fit_n_points: number;
  displacement_2035_pct: { by_scenario: HeadlineScenarioEntry[] };
  base_peak_p50: number;
  mythos_peak_p50: number;
  mythos_delta_vs_base_pp: number;
  sector_snapshot_peak_year: HeadlineSectorSnapshot;
  vectorization_speedup: HeadlineVectorization;
  validation: HeadlineValidation;
}

export type BenchmarkKey = "swe_bench" | "humaneval" | "mmlu";

export interface BenchmarkPoint {
  year: number;
  score: number;
  score_norm: number;
  model: string;
  organization: string;
  benchmark: BenchmarkKey;
  source: string;
  notes: string;
}

export interface BenchmarkFit {
  model: string;
  L: number;
  k: number;
  x0: number;
  ci_95: { L: [number, number]; k: [number, number]; x0: [number, number] };
  r_squared: number;
  rmse: number;
  n_points: number;
  inflection_year: number;
  year_to_reach_99: number | null;
  ci_grid: { year: number; low: number; mid: number; high: number }[];
}

export interface BenchmarksData {
  data_version: string;
  benchmark: BenchmarkKey;
  benchmark_label: string;
  series: BenchmarkPoint[];
  fit: BenchmarkFit;
}

export interface ScenarioSeries {
  key: ScenarioKey;
  name: string;
  label: string;
  description: string;
  color: string;
  narrative: string;
  policy_lever: string;
  mean: number[];
  p05: number[];
  p25: number[];
  p50: number[];
  p75: number[];
  p95: number[];
}

export interface ScenariosData {
  data_version: string;
  n_samples: number;
  seed: number;
  years: number[];
  scenarios: ScenarioSeries[];
}

export interface SectorMetaEntry {
  name: string;
  employment_2025_M: number;
  risk_mean: number;
  risk_std: number;
  direct_replace_frac: number;
  source: string;
}

export interface SectorSnapshotEntry {
  sector: string;
  employment_2025_M: number;
  automation_risk_pct: number;
  at_risk_jobs_M: number;
  displaced_jobs_M: number;
  displacement_pct: number;
  source: string;
}

export interface SectorImpactData {
  data_version: string;
  scenario: string;
  peak_year: number;
  sectors: string[];
  sector_meta: SectorMetaEntry[];
  years: number[];
  matrix: number[][];
  snapshot_peak_year: SectorSnapshotEntry[];
}

export interface ValidationPrediction {
  year: number;
  model: string;
  actual_pct: number;
  predicted_pct: number;
  ci_low_pct: number;
  ci_high_pct: number;
  in_ci_95: boolean;
}

export interface ValidationFold {
  cutoff_year: number;
  n_train: number;
  n_test: number;
  fit: {
    L: number;
    k: number;
    x0: number;
    r_squared: number;
    rmse: number;
  };
  predictions: ValidationPrediction[];
  coverage_95: number;
}

export interface ValidationData {
  data_version: string;
  method: string;
  n_total_points: number;
  folds: ValidationFold[];
  leave_last_out_mythos: ValidationFold;
}

export interface InsightsScenarioEntry {
  headline: string;
  narrative: string;
  policy_lever: string;
}

export interface InsightsData {
  generated_at: string;
  data_version: string;
  note: string;
  executive_summary: string;
  scenarios: Record<ScenarioKey, InsightsScenarioEntry>;
}
