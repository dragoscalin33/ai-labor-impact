import { SectionHeading } from "@/components/site/section-heading";
import { getHeadline, getSectorImpact } from "@/lib/data";

const SNAPSHOT_YEARS = [2025, 2030, 2035, 2040, 2045];

// Sequential single-hue indigo ramp, 6 stops. Saturation capped so the highest
// step never collapses to pure foreground / solid black.
const RAMP_STOPS: string[] = [
  "oklch(0.97 0.01 270)",
  "oklch(0.90 0.04 270)",
  "oklch(0.78 0.08 270)",
  "oklch(0.62 0.12 270)",
  "oklch(0.47 0.14 270)",
  "oklch(0.34 0.13 270)",
];

const RAMP_BREAKS: number[] = [0, 10, 20, 30, 40, 50];

function rampColor(value: number): string {
  for (let i = RAMP_BREAKS.length - 1; i >= 0; i -= 1) {
    if (value >= RAMP_BREAKS[i]) {
      return RAMP_STOPS[i];
    }
  }
  return RAMP_STOPS[0];
}

function rampTextColor(value: number): string {
  return value >= 40 ? "oklch(0.985 0 0)" : "var(--foreground)";
}

export async function SectorHeatmapSection() {
  const [data, headline] = await Promise.all([
    getSectorImpact(),
    getHeadline(),
  ]);
  const yearIndex = (y: number): number => data.years.indexOf(y);

  const rows = data.sectors.map((sector, sIdx) => {
    const row = SNAPSHOT_YEARS.map((y) => {
      const idx = yearIndex(y);
      return idx >= 0 ? data.matrix[sIdx][idx] : 0;
    });
    return { sector, values: row };
  });

  const topSectorName = headline.sector_snapshot_peak_year.top_sector;
  const topSectorPct = headline.sector_snapshot_peak_year.top_sector_displacement_pct;
  const educationHealthcareLast = rows.find(
    (r) => r.sector === "Education & Healthcare"
  )?.values.at(-1);

  return (
    <section
      id="sectors"
      className="border-t border-border/60 bg-muted/30 px-6 py-24"
    >
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="teal"
          eyebrow="Sector impact"
          title="Where displacement concentrates, decade by decade."
          description="Mean displacement percentage under the base scenario, projected per sector and decade. Driven by the fitted sigmoid times per-sector mean risk and direct-replace fraction, net of mitigation. Sectors are ordered by 2025 employment weight; values saturate as the capability curve plateaus."
        />

        <p className="mb-4 inline-flex items-center gap-2 rounded-md border border-border/70 bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground">
          <span className="inline-block size-1.5 rounded-full bg-foreground" />
          Showing: Base scenario (sector heatmap is not re-keyed by the scenario picker)
        </p>

        <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
          <span className="font-medium uppercase tracking-[0.12em]">Scale</span>
          {RAMP_STOPS.map((color, i) => (
            <span key={i} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block size-3 rounded-sm border border-border/40"
                style={{ backgroundColor: color }}
              />
              <span className="tabular-nums">
                {i === RAMP_STOPS.length - 1
                  ? `${RAMP_BREAKS[i]} %+`
                  : `${RAMP_BREAKS[i]}–${RAMP_BREAKS[i + 1]} %`}
              </span>
            </span>
          ))}
        </div>

        <div className="overflow-x-auto rounded-xl border border-border/70 bg-card">
          <div className="min-w-[680px] p-4">
            <div
              className="grid items-center gap-y-1 text-sm"
              style={{ gridTemplateColumns: `minmax(220px, 1.4fr) repeat(${SNAPSHOT_YEARS.length}, minmax(90px, 1fr))` }}
            >
              <div className="text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground">
                Sector
              </div>
              {SNAPSHOT_YEARS.map((y) => (
                <div
                  key={y}
                  className="text-center text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground"
                >
                  {y}
                </div>
              ))}

              {rows.map((row) => (
                <div key={row.sector} className="contents">
                  <div className="pr-3 text-sm text-foreground">{row.sector}</div>
                  {row.values.map((v, i) => (
                    <div
                      key={i}
                      title={`${row.sector} · ${SNAPSHOT_YEARS[i]} · ${v.toFixed(1)} % displacement`}
                      className="m-0.5 flex h-10 items-center justify-center rounded-md text-xs font-medium tabular-nums"
                      style={{
                        backgroundColor: rampColor(v),
                        color: rampTextColor(v),
                      }}
                    >
                      {v.toFixed(1)}%
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        <p className="mt-6 text-sm text-muted-foreground">
          Snapshot at peak year {data.peak_year}: top sector ={" "}
          <span className="text-foreground">{topSectorName}</span> at{" "}
          <span className="text-foreground">{topSectorPct.toFixed(1)} %</span>
          {educationHealthcareLast !== undefined ? (
            <>
              . Education & Healthcare remains the lowest exposed sector at{" "}
              {educationHealthcareLast.toFixed(1)} % displacement
            </>
          ) : null}
          . Source: McKinsey 2023, WEF Future of Jobs 2023, OECD 2024.
        </p>
      </div>
    </section>
  );
}
