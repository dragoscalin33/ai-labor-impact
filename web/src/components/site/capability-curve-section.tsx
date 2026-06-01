import { CapabilityCurveChart } from "@/components/site/capability-curve-chart";
import { SectionHeading } from "@/components/site/section-heading";
import { getBenchmarks } from "@/lib/data";
import { GITHUB_BLOB } from "@/lib/data";

export async function CapabilityCurveSection() {
  const data = await getBenchmarks();

  return (
    <section id="curve" className="border-t border-border/60 bg-muted/30 px-6 py-24">
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="violet"
          eyebrow="Capability curve"
          title="A sigmoid fitted to public AI benchmark scores."
          description={`Logistic fit to ${data.fit.n_points} SWE-bench Verified observations via bounded non-linear least squares. R² = ${data.fit.r_squared.toFixed(
            3
          )}, RMSE = ${data.fit.rmse.toFixed(
            3
          )}. Confidence band derived from the parameter covariance, never from re-fitting on perturbed data.`}
        />

        <CapabilityCurveChart data={data} />

        <p className="mt-6 text-sm text-muted-foreground">
          Inflection year {data.fit.inflection_year.toFixed(2)}. Steepness k ={" "}
          {data.fit.k.toFixed(2)}. Source:{" "}
          <a
            className="text-foreground underline-offset-4 hover:underline"
            href={`${GITHUB_BLOB}/src/models/benchmark_curve.py`}
            target="_blank"
            rel="noreferrer noopener"
          >
            src/models/benchmark_curve.py
          </a>
          .
        </p>
      </div>
    </section>
  );
}
