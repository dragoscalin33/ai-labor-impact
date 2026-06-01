"use client";

import {
  CartesianGrid,
  ErrorBar,
  Label,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ChartMount } from "@/components/site/chart-mount";
import type { ValidationData, ValidationPrediction } from "@/lib/types";

interface Row {
  cutoff_year: number;
  predicted_pct: number;
  actual_pct: number;
  ci_low_pct: number;
  ci_high_pct: number;
  in_ci_95: boolean;
  model: string;
  errorY: [number, number];
}

interface Props {
  data: ValidationData;
}

function buildRows(data: ValidationData): Row[] {
  const rows: Row[] = [];
  for (const fold of data.folds) {
    for (const p of fold.predictions as ValidationPrediction[]) {
      rows.push({
        cutoff_year: Number(fold.cutoff_year.toFixed(2)),
        predicted_pct: Number(p.predicted_pct.toFixed(2)),
        actual_pct: Number(p.actual_pct.toFixed(2)),
        ci_low_pct: Number(p.ci_low_pct.toFixed(2)),
        ci_high_pct: Number(p.ci_high_pct.toFixed(2)),
        in_ci_95: p.in_ci_95,
        model: p.model,
        errorY: [
          Number((p.predicted_pct - p.ci_low_pct).toFixed(2)),
          Number((p.ci_high_pct - p.predicted_pct).toFixed(2)),
        ],
      });
    }
  }
  return rows;
}

interface TooltipPayloadItem {
  payload: Row;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadItem[];
}

function ChartTooltip({ active, payload }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-md border border-border bg-background/95 px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-foreground">{row.model}</div>
      <div className="mt-1 text-muted-foreground">
        Cutoff {row.cutoff_year.toFixed(2)}
      </div>
      <div className="text-muted-foreground">
        Predicted{" "}
        <span className="text-foreground">{row.predicted_pct.toFixed(1)} %</span>{" "}
        · CI [{row.ci_low_pct.toFixed(1)}, {row.ci_high_pct.toFixed(1)}]
      </div>
      <div className="text-muted-foreground">
        Actual <span className="text-foreground">{row.actual_pct.toFixed(1)} %</span>
      </div>
      <div className={row.in_ci_95 ? "text-foreground" : "text-destructive"}>
        {row.in_ci_95 ? "Inside 95 % CI" : "Outside 95 % CI"}
      </div>
    </div>
  );
}

export function ValidationChart({ data }: Props) {
  const rows = buildRows(data);
  const inCi = rows.filter((r) => r.in_ci_95);
  const outCi = rows.filter((r) => !r.in_ci_95);
  const actuals = rows.map((r) => ({
    cutoff_year: r.cutoff_year,
    actual_pct: r.actual_pct,
    model: r.model,
    predicted_pct: r.actual_pct,
    ci_low_pct: r.actual_pct,
    ci_high_pct: r.actual_pct,
    in_ci_95: r.in_ci_95,
    errorY: [0, 0] as [number, number],
  }));

  const totalInCi = inCi.length;
  const totalOutCi = outCi.length;
  const captionId = "validation-chart-caption";

  return (
    <div className="w-full">
      <p id={captionId} className="sr-only">
        Progressive leave-last-out cross-validation results. Each predicted
        point with its 95 percent confidence interval as an error bar, the
        actual observation marked separately. {totalInCi} of {totalInCi + totalOutCi}{" "}
        held-out predictions fall inside the 95 percent confidence interval.
      </p>
      <ChartMount
        className="h-[420px] w-full sm:h-[460px]"
        role="img"
        ariaLabel="Temporal cross-validation chart"
        ariaDescribedBy={captionId}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 24, left: 8, bottom: 32 }}>
            <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="cutoff_year"
              domain={[2024.4, 2025.4]}
              ticks={[2024.6, 2024.75, 2024.83, 2025.1, 2025.15]}
              tickFormatter={(v) => Number(v).toFixed(2)}
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--chart-axis)", fontSize: 12 }}
              tickLine={false}
            >
              <Label
                value="Cutoff year"
                position="insideBottom"
                offset={-12}
                fill="var(--chart-axis)"
                fontSize={12}
              />
            </XAxis>
            <YAxis
              type="number"
              dataKey="predicted_pct"
              domain={[0, 105]}
              stroke="var(--chart-axis)"
              tick={{ fill: "var(--chart-axis)", fontSize: 12 }}
              tickLine={false}
              tickFormatter={(v) => `${v}%`}
              width={48}
            >
              <Label
                value="SWE-bench score"
                position="insideLeft"
                angle={-90}
                offset={12}
                style={{ textAnchor: "middle" }}
                fill="var(--chart-axis)"
                fontSize={12}
              />
            </YAxis>
            <ReferenceLine
              y={93.9}
              stroke="var(--chart-axis)"
              strokeDasharray="4 4"
              label={{
                value: "Mythos 93.9 %",
                position: "insideTopRight",
                fill: "var(--chart-axis)",
                fontSize: 11,
              }}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--chart-axis)" }} />
            <Scatter data={inCi} fill="var(--chart-5)" name="Predicted (inside 95 % CI)">
              <ErrorBar dataKey="errorY" stroke="var(--chart-5)" strokeWidth={1.2} width={4} />
            </Scatter>
            <Scatter
              data={outCi}
              fill="var(--destructive)"
              fillOpacity={0.7}
              name="Predicted (outside 95 % CI)"
            >
              <ErrorBar
                dataKey="errorY"
                stroke="var(--destructive)"
                strokeOpacity={0.6}
                strokeWidth={1.2}
                width={4}
              />
            </Scatter>
            <Scatter
              data={actuals}
              dataKey="actual_pct"
              fill="var(--foreground)"
              shape="cross"
              name="Actual"
            />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartMount>
      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rounded-full bg-[var(--chart-5)]" />
          Predicted, inside 95 % CI
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rounded-full bg-[var(--destructive)] opacity-70" />
          Predicted, outside 95 % CI
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-block size-2 rotate-45 bg-foreground" />
          Actual observation
        </span>
      </div>
    </div>
  );
}
