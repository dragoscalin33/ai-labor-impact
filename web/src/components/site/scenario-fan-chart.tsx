"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Label,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ChartMount } from "@/components/site/chart-mount";
import type { ScenarioSeries } from "@/lib/types";

interface Row {
  year: number;
  median: number;
  p25: number;
  p75: number;
  p05: number;
  p95: number;
  band_inner: [number, number];
  band_outer: [number, number];
}

interface Props {
  series: ScenarioSeries;
  years: number[];
}

function makeRows(series: ScenarioSeries, years: number[]): Row[] {
  return years.map((year, i) => ({
    year,
    median: Number(series.p50[i].toFixed(2)),
    p25: Number(series.p25[i].toFixed(2)),
    p75: Number(series.p75[i].toFixed(2)),
    p05: Number(series.p05[i].toFixed(2)),
    p95: Number(series.p95[i].toFixed(2)),
    band_inner: [Number(series.p25[i].toFixed(2)), Number(series.p75[i].toFixed(2))],
    band_outer: [Number(series.p05[i].toFixed(2)), Number(series.p95[i].toFixed(2))],
  }));
}

interface TooltipPayloadItem {
  dataKey: string;
  payload: Row;
  value: number;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}

function ChartTooltip({ active, payload, label }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-md border border-border bg-background/95 px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-foreground">Year {label}</div>
      <div className="mt-1 text-muted-foreground">
        Median <span className="text-foreground">{row.median.toFixed(1)} %</span>
      </div>
      <div className="text-muted-foreground">
        P25–P75 {row.p25.toFixed(1)}–{row.p75.toFixed(1)} %
      </div>
      <div className="text-muted-foreground">
        P05–P95 {row.p05.toFixed(1)}–{row.p95.toFixed(1)} %
      </div>
    </div>
  );
}

export function ScenarioFanChart({ series, years }: Props) {
  const rows = makeRows(series, years);
  const captionId = `fan-chart-caption-${series.key}`;
  const last = rows[rows.length - 1];
  const peakIdx = rows.reduce(
    (best, r, i) => (r.median > rows[best].median ? i : best),
    0
  );
  const peak = rows[peakIdx];

  return (
    <>
      <p id={captionId} className="sr-only">
        Fan chart for {series.name} between {years[0]} and {years[years.length - 1]}.
        Peak median displacement {peak.median.toFixed(1)} percent in {peak.year},
        with P05 to P95 envelope from {peak.p05.toFixed(1)} to {peak.p95.toFixed(1)} percent.
        Final {last.year} median {last.median.toFixed(1)} percent.
      </p>
      <ChartMount
        className="h-[360px] w-full sm:h-[420px]"
        role="img"
        ariaLabel={`${series.name} fan chart`}
        ariaDescribedBy={captionId}
      >
        <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={rows}
          margin={{ top: 16, right: 24, left: 8, bottom: 32 }}
        >
          <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
          <XAxis
            dataKey="year"
            type="number"
            domain={["dataMin", "dataMax"]}
            ticks={[2025, 2030, 2035, 2040, 2045, 2050]}
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
            domain={[0, "dataMax + 5"]}
            stroke="var(--chart-axis)"
            tick={{ fill: "var(--chart-axis)", fontSize: 12 }}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
            width={48}
          >
            <Label
              value="Unemployment"
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
          <Area
            type="monotone"
            dataKey="band_outer"
            stroke="none"
            fill="var(--chart-3)"
            fillOpacity={0.16}
            isAnimationActive={false}
            name="P05–P95"
          />
          <Area
            type="monotone"
            dataKey="band_inner"
            stroke="none"
            fill="var(--chart-3)"
            fillOpacity={0.38}
            isAnimationActive={false}
            name="P25–P75"
          />
          <Line
            type="monotone"
            dataKey="median"
            stroke="var(--chart-5)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            name="Median"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartMount>
    </>
  );
}
