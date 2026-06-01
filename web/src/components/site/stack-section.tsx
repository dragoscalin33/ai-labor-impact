import { SectionHeading } from "@/components/site/section-heading";
import { Badge } from "@/components/ui/badge";
import { GITHUB_BLOB, GITHUB_REPO_URL } from "@/lib/data";

const MODEL_STACK = [
  "Python 3.10–3.12",
  "NumPy",
  "SciPy",
  "PyMC",
  "MLflow",
  "pytest",
  "ruff",
  "mypy",
];

const SITE_STACK = [
  "Next.js 16",
  "React 19",
  "TypeScript strict",
  "Tailwind CSS v4",
  "shadcn/ui",
  "Recharts",
  "Vercel AI SDK",
  "Vercel",
];

interface FileLink {
  path: string;
  note: string;
}

const HIGHLIGHTS: FileLink[] = [
  {
    path: "src/models/scenarios.py",
    note: "Vectorised Monte Carlo simulator",
  },
  {
    path: "src/models/bayesian.py",
    note: "Hierarchical PyMC sector-risk model",
  },
  {
    path: "src/validation/temporal_cv.py",
    note: "Leave-last-out + rolling-origin CV",
  },
  {
    path: "notebooks/05_temporal_validation.ipynb",
    note: "End-to-end validation walkthrough",
  },
];

export function StackSection() {
  return (
    <section
      id="stack"
      className="border-t border-border/60 px-6 py-16"
    >
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="stone"
          eyebrow="Code & stack"
          title="Python engine on one side, Next.js vitrine on the other."
          description="The model code is unchanged from the engine repository; the site reads its precomputed JSON artefacts."
        />

        <div className="grid gap-8 lg:grid-cols-2">
          <article className="rounded-xl border border-border/70 bg-card p-6">
            <h3 className="font-heading text-base font-medium text-foreground">
              Model
            </h3>
            <div className="mt-4 flex flex-wrap gap-2">
              {MODEL_STACK.map((s) => (
                <Badge key={s} variant="outline" className="font-normal">
                  {s}
                </Badge>
              ))}
            </div>
            <p className="mt-4 text-xs text-muted-foreground">
              Continuous integration runs ruff, mypy and pytest on Python 3.10,
              3.11 and 3.12.
            </p>
            <p className="mt-3">
              <a
                href={`${GITHUB_REPO_URL}/actions/workflows/ci.yml`}
                target="_blank"
                rel="noreferrer noopener"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`${GITHUB_REPO_URL}/actions/workflows/ci.yml/badge.svg`}
                  alt="CI status badge"
                  className="h-5"
                />
              </a>
            </p>
          </article>

          <article className="rounded-xl border border-border/70 bg-card p-6">
            <h3 className="font-heading text-base font-medium text-foreground">
              Site
            </h3>
            <div className="mt-4 flex flex-wrap gap-2">
              {SITE_STACK.map((s) => (
                <Badge key={s} variant="outline" className="font-normal">
                  {s}
                </Badge>
              ))}
            </div>
            <p className="mt-4 text-xs text-muted-foreground">
              Static rendering, no client-side data fetching. Charts render in
              client components only where Recharts requires the browser DOM.
            </p>
          </article>
        </div>

        <div className="mt-12">
          <h3 className="font-heading text-base font-medium text-foreground">
            Highlight files
          </h3>
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {HIGHLIGHTS.map((file) => (
              <li
                key={file.path}
                className="rounded-md border border-border/60 bg-card px-4 py-3"
              >
                <a
                  href={`${GITHUB_BLOB}/${file.path}`}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="block text-sm font-medium text-foreground underline-offset-4 hover:underline"
                >
                  {file.path}
                </a>
                <p className="text-xs text-muted-foreground">{file.note}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
