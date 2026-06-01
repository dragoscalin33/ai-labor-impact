import Link from "next/link";

import { Button } from "@/components/ui/button";
import { getHeadline, GITHUB_REPO_URL } from "@/lib/data";

const DATA_SOURCES = [
  { label: "World Bank Open Data", note: "Global unemployment & employment, 1991–2024" },
  { label: "Papers with Code", note: "SWE-bench, HumanEval, MMLU scores" },
  { label: "McKinsey / WEF / OECD", note: "Sector automation risk estimates" },
  { label: "Anthropic red team", note: "Mythos Preview SWE-bench 93.9 %" },
];

export async function HeroSection() {
  const headline = await getHeadline();
  const base = headline.displacement_2035_pct.by_scenario.find(
    (s) => s.key === "base"
  );

  return (
    <section
      id="top"
      className="relative px-6 pt-20 pb-16"
    >
      <div className="mx-auto w-full max-w-[1080px]">
        <p className="mb-6 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
          AI labor market impact observatory
        </p>
        <h1 className="font-heading text-4xl font-medium tracking-tight text-foreground sm:text-5xl md:text-6xl">
          Modelling how AI capability translates into labour displacement,
          <span className="text-muted-foreground"> with uncertainty you can audit.</span>
        </h1>

        {base ? (
          <div className="mt-10 flex flex-col gap-2 border-l-2 border-foreground/60 pl-5">
            <p className="font-heading text-5xl font-medium tabular-nums tracking-tight text-foreground sm:text-6xl md:text-[5rem]">
              {base.p05_2035.toFixed(1)}–{base.p95_2035.toFixed(1)} %
            </p>
            <p className="text-sm text-muted-foreground sm:text-base">
              Base scenario, 2035, 95 % Monte Carlo interval (n = {headline.n_samples.toLocaleString()}).
              Median {base.p50_2035.toFixed(1)} %, peak {base.peak_p50.toFixed(1)} % in {base.peak_year}.
            </p>
          </div>
        ) : null}

        <p className="mt-8 max-w-3xl text-base leading-relaxed text-muted-foreground sm:text-lg">
          Every parameter fitted to a public data source. Every projection a
          vectorised Monte Carlo distribution over five thousand trials.
          Predictive credibility audited with leave-last-out and rolling-origin
          temporal cross-validation.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <Button asChild size="lg">
            <a href="#findings">See the model</a>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noreferrer noopener"
            >
              View source
            </Link>
          </Button>
        </div>

        <div className="mt-16 border-t border-border/70 pt-8">
          <p className="text-xs font-medium uppercase tracking-[0.16em] text-muted-foreground">
            Verified data sources
          </p>
          <ul className="mt-4 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            {DATA_SOURCES.map((src) => (
              <li key={src.label} className="text-foreground">
                <span className="block font-medium">{src.label}</span>
                <span className="block text-xs leading-snug text-muted-foreground">
                  {src.note}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
