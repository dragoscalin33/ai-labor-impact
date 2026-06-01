import { Card, CardContent } from "@/components/ui/card";
import { getHeadline } from "@/lib/data";

function formatPercentRange(p05: number, p95: number): string {
  return `${p05.toFixed(1)}–${p95.toFixed(1)} %`;
}

export async function KeyFindingsSection() {
  const headline = await getHeadline();
  const base = headline.displacement_2035_pct.by_scenario.find(
    (s) => s.key === "base"
  );

  const cards: { value: string; label: string; note: string }[] = [
    {
      value: `${(headline.vectorization_speedup.vectorized_seconds * 1000).toFixed(0)} ms`,
      label: "Monte Carlo run, vectorised",
      note: `n = ${headline.n_samples.toLocaleString()} trials × 26 years × 11 sectors, ${headline.vectorization_speedup.speedup_factor.toFixed(
        0
      )}× faster than the naive loop.`,
    },
    {
      value: `${headline.swe_bench_mythos.score.toFixed(1)} %`,
      label: "Claude Mythos Preview on SWE-bench Verified",
      note: `April 2026. Inside the 95 % CI of the leave-last-out sigmoid (8 train, 1 held out).`,
    },
    {
      value: base
        ? formatPercentRange(base.p05_2035, base.p95_2035)
        : "—",
      label: "Base-scenario displacement at 2035 (95 % MC interval)",
      note: base
        ? `Median ${base.p50_2035.toFixed(1)} %. Peak ${base.peak_p50.toFixed(
            1
          )} % at ${base.peak_year}.`
        : "",
    },
    {
      value: `${headline.fit_r_squared.toFixed(3)}`,
      label: "Sigmoid R² on SWE-bench Verified",
      note: `${headline.fit_n_points} public scores, inflection ${headline.fit_inflection_year.toFixed(
        1
      )}, CIs from parameter covariance.`,
    },
  ];

  return (
    <section
      id="findings"
      className="px-6 py-24"
    >
      <div className="mx-auto w-full max-w-[1080px]">
        <p className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
          Key findings
        </p>
        <h2 className="font-heading mt-3 max-w-3xl text-3xl font-medium tracking-tight text-foreground sm:text-[2rem]">
          Four numbers that anchor the rest of the model.
        </h2>
        <p className="mt-4 max-w-3xl text-base leading-relaxed text-muted-foreground">
          All values regenerated from{" "}
          <code className="rounded bg-muted px-1.5 py-0.5 text-[0.85em] text-foreground">
            scripts/precompute_web_artifacts.py
          </code>{" "}
          with seed&nbsp;{headline.seed}, data version {headline.data_version}.
        </p>

        <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {cards.map((card) => (
            <Card key={card.label} className="h-full">
              <CardContent className="flex h-full flex-col gap-2 py-2">
                <div className="font-heading whitespace-nowrap text-[1.75rem] font-medium tracking-tight tabular-nums text-foreground sm:text-[1.875rem] lg:text-[2rem] xl:text-[2.25rem]">
                  {card.value}
                </div>
                <div className="text-sm font-medium text-foreground">
                  {card.label}
                </div>
                <p className="mt-auto text-xs leading-relaxed text-muted-foreground">
                  {card.note}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
