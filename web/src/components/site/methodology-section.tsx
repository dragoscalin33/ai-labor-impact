import { SectionHeading } from "@/components/site/section-heading";
import { GITHUB_BLOB } from "@/lib/data";

interface Block {
  title: string;
  body: string;
  source: string;
  href: string;
}

const BLOCKS: Block[] = [
  {
    title: "Capability curve",
    body: "A logistic function f(t) = L / (1 + exp(-k (t − t₀))) is fitted to public SWE-bench, HumanEval and MMLU scores via bounded non-linear least squares (Levenberg–Marquardt). 95 % confidence intervals are derived from the parameter covariance via the Student-t distribution. Predictive uncertainty is propagated by sampling (L, k, t₀) from the multivariate normal of the fit — never by re-fitting on perturbed data.",
    source: "src/models/benchmark_curve.py",
    href: `${GITHUB_BLOB}/src/models/benchmark_curve.py`,
  },
  {
    title: "Monte Carlo simulator",
    body: "For each of n = 5,000 trials, sigmoid parameters are drawn from the fit covariance, sector automation risks are sampled from per-sector truncated normals, and displacement is computed as automation × risk × direct-replace × (1 − mitigation) per sector, per year, net of new-job creation. The whole loop is a single NumPy broadcast over (samples × years × sectors).",
    source: "src/models/scenarios.py",
    href: `${GITHUB_BLOB}/src/models/scenarios.py`,
  },
  {
    title: "Hierarchical Bayesian sector risk",
    body: "A partial-pooling PyMC model with a non-centred parameterisation: μ_g ~ Normal(0.5, 0.2), τ ~ HalfNormal(0.15), z_s ~ Normal(0, 1), θ_s = clip(μ_g + τ z_s, 0.01, 0.99). The posterior is a drop-in replacement for the truncated-normal draws in the Monte Carlo simulator.",
    source: "src/models/bayesian.py",
    href: `${GITHUB_BLOB}/src/models/bayesian.py`,
  },
  {
    title: "Temporal cross-validation",
    body: "Two protocols: leave-last-out (for each observation i, fit on 0..i−1 and predict i with its 95 % CI) and rolling-origin (at each cutoff date, forecast h ∈ {0.5, 1.0, 2.0} years ahead). Standard discipline from operational time-series forecasting (Hyndman & Athanasopoulos, FPP3 ch. 5).",
    source: "src/validation/temporal_cv.py",
    href: `${GITHUB_BLOB}/src/validation/temporal_cv.py`,
  },
  {
    title: "Known limitations",
    body: "Capability is proxied by a single bounded benchmark (SWE-bench, 0–100 %). The fitted sigmoid already has its inflection year at 2024.67 and is close to its asymptote, so all four scenarios converge against the same ceiling — the spread between Base, Pessimistic and Mythos-accelerated is one percentage point at peak. That is the cost of using a saturating benchmark as the capability axis; once the benchmark saturates, the model underestimates the right tail. v2 replaces the single-benchmark sigmoid with a chained-benchmark envelope (SWE-bench → SWE-bench Hard → task-completion-rate). Until then, read every projection here as correct on validation, conservative on tail risk, and explicit about why.",
    source: "src/models/scenarios.py",
    href: `${GITHUB_BLOB}/src/models/scenarios.py`,
  },
];

export function MethodologySection() {
  return (
    <section id="methodology" className="border-t border-border/60 bg-muted/30 px-6 py-16">
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="sky"
          eyebrow="Methodology"
          title="Four modelling blocks, each auditable in isolation."
          description="Each artefact on this page is regenerated from one of these blocks. The Python engine is unchanged; the site is a thin presentation layer over its JSON output."
        />

        <div className="grid gap-8 md:grid-cols-2">
          {BLOCKS.map((b) => (
            <article key={b.title} className="rounded-xl border border-border/70 bg-card p-6">
              <h3 className="font-heading text-lg font-medium tracking-tight text-foreground">
                {b.title}
              </h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {b.body}
              </p>
              <p className="mt-4 text-xs text-muted-foreground">
                Source:{" "}
                <a
                  className="text-foreground underline-offset-4 hover:underline"
                  href={b.href}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  {b.source}
                </a>
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
