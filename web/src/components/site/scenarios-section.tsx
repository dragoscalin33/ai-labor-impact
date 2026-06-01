import { Fragment } from "react";

import { ScenarioFanChart } from "@/components/site/scenario-fan-chart";
import { ScenariosTabs } from "@/components/site/scenarios-tabs";
import { SectionHeading } from "@/components/site/section-heading";
import { TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getHeadline, getScenarios } from "@/lib/data";
import type { ScenarioKey, ScenarioSeries } from "@/lib/types";

function renderInline(text: string, baseKey: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const boldRe = /\*\*([^*]+)\*\*/g;
  let cursor = 0;
  let match: RegExpExecArray | null;
  let idx = 0;
  while ((match = boldRe.exec(text)) !== null) {
    if (match.index > cursor) {
      nodes.push(renderItalic(text.slice(cursor, match.index), `${baseKey}-t${idx++}`));
    }
    nodes.push(
      <strong key={`${baseKey}-b${idx++}`} className="font-medium text-foreground">
        {match[1]}
      </strong>
    );
    cursor = boldRe.lastIndex;
  }
  if (cursor < text.length) {
    nodes.push(renderItalic(text.slice(cursor), `${baseKey}-t${idx++}`));
  }
  return nodes;
}

function renderItalic(text: string, baseKey: string): React.ReactNode {
  const italicRe = /\*([^*]+)\*/g;
  const out: React.ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  let idx = 0;
  while ((match = italicRe.exec(text)) !== null) {
    if (match.index > cursor) {
      out.push(<Fragment key={`${baseKey}-r${idx++}`}>{text.slice(cursor, match.index)}</Fragment>);
    }
    out.push(
      <em key={`${baseKey}-i${idx++}`} className="italic">
        {match[1]}
      </em>
    );
    cursor = italicRe.lastIndex;
  }
  if (cursor < text.length) {
    out.push(<Fragment key={`${baseKey}-r${idx++}`}>{text.slice(cursor)}</Fragment>);
  }
  return <>{out}</>;
}

const SHORT_LABEL: Record<ScenarioKey, string> = {
  optimistic: "Optimistic",
  base: "Base",
  pessimistic: "Pessimistic",
  mythos_accelerated: "Mythos",
};

const TAB_SUBTITLE: Record<ScenarioKey, string> = {
  optimistic: "managed transition",
  base: "no intervention",
  pessimistic: "structural collapse",
  mythos_accelerated: "cyber cascade",
};

const ORDER: ScenarioKey[] = [
  "optimistic",
  "base",
  "pessimistic",
  "mythos_accelerated",
];

interface KpiTileProps {
  label: string;
  value: string;
  sub: string;
}

function KpiTile({ label, value, sub }: KpiTileProps) {
  return (
    <div className="rounded-lg border border-border/70 bg-card px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </p>
      <p className="font-heading mt-1 text-2xl font-medium text-foreground">
        {value}
      </p>
      <p className="text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

interface ScenarioPanelProps {
  series: ScenarioSeries;
  years: number[];
  peakInfo: { year: number; p50: number; p05: number; p95: number };
}

function ScenarioPanel({ series, years, peakInfo }: ScenarioPanelProps) {
  const paragraphs = series.narrative.split(/\n+/).filter((p) => p.trim().length > 0);
  return (
    <div className="space-y-8">
      <ScenarioFanChart series={series} years={years} />
      <div className="grid gap-8 lg:grid-cols-[240px_1fr]">
        <div className="flex flex-col gap-3">
          <KpiTile
            label="Peak year"
            value={String(peakInfo.year)}
            sub="median maximum"
          />
          <KpiTile
            label="Peak (P05–P95)"
            value={`${peakInfo.p05.toFixed(1)}–${peakInfo.p95.toFixed(1)} %`}
            sub={`median ${peakInfo.p50.toFixed(1)} %`}
          />
        </div>
        <div className="max-w-[640px] text-sm leading-relaxed text-muted-foreground">
          {paragraphs.map((para, i) => (
            <p key={i} className="mb-3 last:mb-0">
              {renderInline(para, `p${i}`)}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

export async function ScenariosSection() {
  const [scenarios, headline] = await Promise.all([getScenarios(), getHeadline()]);

  const peakByKey = new Map<ScenarioKey, { year: number; p50: number; p05: number; p95: number }>();
  for (const entry of headline.displacement_2035_pct.by_scenario) {
    const ser = scenarios.scenarios.find((s) => s.key === entry.key);
    if (!ser) continue;
    const idx = scenarios.years.indexOf(entry.peak_year);
    peakByKey.set(entry.key, {
      year: entry.peak_year,
      p50: entry.peak_p50,
      p05: idx >= 0 ? ser.p05[idx] : entry.peak_p50,
      p95: idx >= 0 ? ser.p95[idx] : entry.peak_p50,
    });
  }

  return (
    <section id="scenarios" className="border-t border-border/60 px-6 py-24">
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="amber"
          eyebrow="Scenarios"
          title="Four policy worlds, with full Monte Carlo uncertainty."
          description={`Each scenario draws ${scenarios.n_samples.toLocaleString()} parameter samples from the fit covariance and from sector-risk truncated normals calibrated to McKinsey, WEF and OECD. The shaded bands are the P25–P75 (inner) and P05–P95 (outer) Monte Carlo intervals; the line is the median.`}
        />

        <ScenariosTabs>
          <TabsList className="mb-8 h-auto w-full">
            {ORDER.map((key) => {
              const peak = peakByKey.get(key);
              return (
                <TabsTrigger
                  key={key}
                  value={key}
                  className="h-auto flex-col items-start gap-0.5 px-3 py-2.5 text-left"
                >
                  <span className="text-sm font-medium">
                    {SHORT_LABEL[key]}
                    <span className="ml-2 hidden text-[0.7rem] font-normal uppercase tracking-[0.08em] text-muted-foreground sm:inline">
                      {TAB_SUBTITLE[key]}
                    </span>
                  </span>
                  {peak ? (
                    <span className="text-[0.7rem] tabular-nums text-muted-foreground">
                      peak {peak.p05.toFixed(1)}–{peak.p95.toFixed(1)} % · {peak.year}
                    </span>
                  ) : null}
                </TabsTrigger>
              );
            })}
          </TabsList>

          {ORDER.map((key) => {
            const series = scenarios.scenarios.find((s) => s.key === key);
            const peak = peakByKey.get(key);
            if (!series || !peak) return null;
            return (
              <TabsContent key={key} value={key}>
                <ScenarioPanel series={series} years={scenarios.years} peakInfo={peak} />
              </TabsContent>
            );
          })}
        </ScenariosTabs>
      </div>
    </section>
  );
}
