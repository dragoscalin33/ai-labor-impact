import { SectionHeading } from "@/components/site/section-heading";
import { ValidationChart } from "@/components/site/validation-chart";
import { GITHUB_BLOB, getValidation } from "@/lib/data";

export async function ValidationSection() {
  const data = await getValidation();
  const mythosFold = data.leave_last_out_mythos;
  const mythosPred = mythosFold.predictions[0];

  return (
    <section id="validation" className="border-t border-border/60 px-6 py-16">
      <div className="mx-auto w-full max-w-[1080px]">
        <SectionHeading
          accent="emerald"
          eyebrow="Validation"
          title="Out-of-sample, not just R² on the training data."
          description="Progressive leave-last-out temporal cross-validation on the SWE-bench Verified series. At each cutoff, the sigmoid is fitted exclusively on observations published before that date; the held-out points are then projected with their 95 % confidence interval."
        />

        <ValidationChart data={data} />

        <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_320px]">
          <blockquote className="rounded-lg border-l-2 border-foreground/40 bg-muted/40 px-5 py-4 text-base leading-relaxed text-foreground">
            “The model fitted only on data published before April 2026 placed
            the Claude Mythos Preview observation (
            {mythosPred ? `${mythosPred.actual_pct.toFixed(1)} %` : "93.9 %"}) inside
            its 95 % confidence interval — predicted{" "}
            {mythosPred ? `${mythosPred.predicted_pct.toFixed(1)} %` : ""}, CI [
            {mythosPred?.ci_low_pct.toFixed(1) ?? ""},{" "}
            {mythosPred?.ci_high_pct.toFixed(1) ?? ""}].”
          </blockquote>
          <div className="text-sm leading-relaxed text-muted-foreground">
            <p>
              The protocol is standard time-series discipline (Hyndman &
              Athanasopoulos, FPP3, ch. 5). Folds with few training points
              produce wide CIs that contain the truth; folds with more points
              produce tighter CIs that earn their keep.
            </p>
            <p className="mt-3">
              Source:{" "}
              <a
                className="text-foreground underline-offset-4 hover:underline"
                href={`${GITHUB_BLOB}/src/validation/temporal_cv.py`}
                target="_blank"
                rel="noreferrer noopener"
              >
                src/validation/temporal_cv.py
              </a>
              ,{" "}
              <a
                className="text-foreground underline-offset-4 hover:underline"
                href={`${GITHUB_BLOB}/notebooks/05_temporal_validation.ipynb`}
                target="_blank"
                rel="noreferrer noopener"
              >
                notebooks/05_temporal_validation.ipynb
              </a>
              .
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
