"use client";

import {
  CartesianGrid,
  ComposedChart,
  Label,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ChartMount } from "@/components/site/chart-mount";
import type { BenchmarksData } from "@/lib/types";

const BENCHMARK_COLORS: Record<string, string> = {
  swe_bench: "var(--chart-5)",
  humaneval: "var(--chart-2)",
  mmlu: "var(--chart-3)",
};

const BENCHMARK_LABELS: Record<string, string> = {
  swe_bench: "SWE-bench",
  humaneval: "HumanEval",
  mmlu: "MMLU",
};

interface Props {
  data: BenchmarksData;
}

interface FitRow {
  year: number;
  fit_low: number;
  fit_mid: number;
  fit_high: number;
}

interface PointRow {
  year: number;
  score_pct: number;
  benchmark: string;
  model: string;
}

function makeFitSeries(data: BenchmarksData): FitRow[] {
  return data.fit.ci_grid
    .filter((p) => p.year >= 2018 && p.year <= 2030)
    .map((p) => ({
      year: Number(p.year.toFixed(2)),
      fit_low: Number((p.low * 100).toFixed(2)),
      fit_mid: Number((p.mid * 100).toFixed(2)),
      fit_high: Number((p.high * 100).toFixed(2)),
    }));
}

function makePointsByBenchmark(data: BenchmarksData): Record<string, PointRow[]> {
  const out: Record<string, PointRow[]> = {
    swe_bench: [],
    humaneval: [],
    mmlu: [],
  };
  for (const p of data.series) {
    if (!out[p.benchmark]) continue;
    out[p.benchmark].push({
      year: p.year,
      score_pct: p.score,
      benchmark: p.benchmark,
      model: p.model,
    });
  }
  return out;
}

interface TooltipPayloadItem {
  dataKey: string;
  payload: PointRow | FitRow;
  value: number;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}

function ChartTooltip({ active, payload, label }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const pointRows = payload.filter(
    (entry): entry is TooltipPayloadItem & { payload: PointRow } =>
      entry.dataKey === "score_pct" && "model" in entry.payload
  );
  const fitRow = payload.find((entry) => entry.dataKey === "fit_mid");

  return (
    <div className="rounded-md border border-border bg-background/95 px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-foreground">
        Year {typeof label === "number" ? label.toFixed(2) : label}
      </div>
      {pointRows.map((row, idx) => {
        const p = row.payload;
        return (
          <div key={`${p.model}-${idx}`} className="mt-1 text-muted-foreground">
            <span className="text-foreground">{p.model}</span> ·{" "}
            {BENCHMARK_LABELS[p.benchmark] ?? p.benchmark} · {p.score_pct.toFixed(1)} %
          </div>
        );
      })}
      {fitRow ? (
        <div className="mt-1 text-muted-foreground">
          SWE-bench fit: {(fitRow.value).toFixed(1)} %
        </div>
      ) : null}
    </div>
  );
}

export function CapabilityCurveChart({ data }: Props) {
  const fitSeries = makeFitSeries(data);
  const pointsByBench = makePointsByBenchmark(data);

  const mythos = data.series.find(
    (p) => p.model === "Claude Mythos Preview" && p.benchmark === "swe_bench"
  );

  const captionId = "capability-curve-caption";

  return (
    <div className="w-full">
      <p id={captionId} className="sr-only">
        Sigmoid fit to SWE-bench Verified observations between 2018 and 2030,
        with a 95 % confidence band, plus HumanEval and MMLU observation
        scatter. Inflection year {data.fit.inflection_year.toFixed(2)}, R²{" "}
        {data.fit.r_squared.toFixed(3)}.
        {mythos
          ? ` Claude Mythos Preview observation at ${mythos.year.toFixed(
              2
            )} scoring ${mythos.score.toFixed(1)} percent.`
          : ""}
      </p>
      <ChartMount
        className="h-[460px] w-full sm:h-[520px]"
        role="img"
        ariaLabel="Capability curve chart"
        ariaDescribedBy={captionId}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={fitSeries}
            margin={{ top: 16, right: 24, left: 8, bottom: 32 }}
          >
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
            <XAxis
              dataKey="year"
              type="number"
              domain={[2018, 2030]}
              ticks={[2018, 2020, 2022, 2024, 2026, 2028, 2030]}
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--chart-axis)", fontSize: 12 }}
              tickLine={false}
            >
              <Label
                value="Year"
                position="insideBottom"
                offset={-12}
                fill="var(--chart-axis)"
                fontSize={12}
              />
            </XAxis>
            <YAxis
              type="number"
              domain={[0, 100]}
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--chart-axis)", fontSize: 12 }}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
              width={48}
            >
              <Label
                value="Benchmark score"
                position="insideLeft"
                angle={-90}
                offset={12}
                style={{ textAnchor: "middle" }}
                fill="var(--chart-axis)"
                fontSize={12}
              />
            </YAxis>
            <Tooltip
              content={<ChartTooltip />}
              cursor={{ stroke: "var(--chart-axis)", strokeDasharray: "2 4" }}
            />
            <Line
              type="monotone"
              dataKey="fit_high"
              stroke="var(--chart-3)"
              strokeWidth={1}
              strokeDasharray="3 3"
              dot={false}
              isAnimationActive={false}
              name="SWE-bench fit (upper 95 % CI)"
            />
            <Line
              type="monotone"
              dataKey="fit_low"
              stroke="var(--chart-3)"
              strokeWidth={1}
              strokeDasharray="3 3"
              dot={false}
              isAnimationActive={false}
              name="SWE-bench fit (lower 95 % CI)"
            />
            <Line
              type="monotone"
              dataKey="fit_mid"
              stroke="var(--chart-5)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="SWE-bench fit (median)"
            />
            {(["swe_bench", "humaneval", "mmlu"] as const).map((bench) => (
              <Scatter
                key={bench}
                data={pointsByBench[bench]}
                dataKey="score_pct"
                fill={BENCHMARK_COLORS[bench]}
                shape={bench === "swe_bench" ? "circle" : "triangle"}
                name={BENCHMARK_LABELS[bench]}
                isAnimationActive={false}
              />
            ))}
            {mythos ? (
              <ReferenceDot
                x={mythos.year}
                y={mythos.score}
                r={6}
                fill="var(--foreground)"
                stroke="var(--background)"
                strokeWidth={2}
              >
                <Label
                  value={`Mythos Preview · ${mythos.score.toFixed(1)} %`}
                  position="top"
                  offset={10}
                  fill="var(--foreground)"
                  fontSize={11}
                />
              </ReferenceDot>
            ) : null}
          </ComposedChart>
        </ResponsiveContainer>
      </ChartMount>
      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-[2px] w-6 bg-[var(--chart-5)]" />
          SWE-bench sigmoid fit (median)
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-[2px] w-6 border-t border-dashed border-[var(--chart-3)]" />
          95 % confidence band
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rounded-full bg-[var(--chart-5)]" />
          SWE-bench observations
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rotate-180 bg-[var(--chart-2)]" />
          HumanEval observations
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rotate-180 bg-[var(--chart-3)]" />
          MMLU observations
        </span>
      </div>
    </div>
  );
}
