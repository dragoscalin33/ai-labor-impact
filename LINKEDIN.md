# LinkedIn launch drafts

## Winner — finding-first::hookA (tightened)

A logistic curve fit on 8 SWE-bench data points predicted Claude Mythos Preview would score 82.4% in April 2026.

Actual: 93.9%. Inside the 95% CI.

That is the leave-last-out test from a project I have been building: a reproducible model of how AI capability progression maps onto sector-by-sector labour displacement.

What is under the hood:

- Sigmoid fit to 9 SWE-bench Verified results from 2021 to 2026. R-squared 0.972, inflection year 2024.67
- Progressive leave-last-out and rolling-origin cross-validation across 5 folds
- Vectorised Monte Carlo, 5000 samples, seed 42, single NumPy broadcast over (samples, years, sectors). 0.026 seconds versus 30 seconds for the triple-nested Python loop. 1153x speedup
- Hierarchical Bayesian PyMC layer on sector susceptibility, non-centred parameterisation, drop-in replacement for the truncated-normal draws
- Four scenarios: optimistic, base, pessimistic, and a Mythos-accelerated cybersecurity cascade

Base case: median peak displacement 25.4% in 2029, 95% interval 21.4 to 27.6 percentage points by 2035.

The Mythos-accelerated path adds under 1 percentage point at peak versus the base case. The catastrophe is not the headline benchmark jump. It is the sector concentration: Administrative Services hits 57.3% by 2040.

Live site: https://ai-labor-impact.vercel.app
Code: https://github.com/dragoscalin33/ai-labor-impact

## Runner-up — methodology-first::hookB

Reporting R² = 0.97 on the training data is not evidence that a forecast is predictive.

The honest test is out-of-sample.

That single sentence shaped most of the design choices in this project: AI Labor Market Impact Observatory, a reproducible model of how AI capability progression maps onto global labour displacement.

What that commitment looks like in code:

— Leave-last-out and rolling-origin temporal cross-validation on the SWE-bench Verified series. For each cutoff, the sigmoid is refit on prior data only, and the held-out observation is compared against the 95 percent CI. Fitting only on data published before April 2026 placed Claude Mythos Preview (93.9 percent) inside the CI.

— A logistic curve fitted by bounded non-linear least squares to SWE-bench, HumanEval and MMLU. Predictive uncertainty propagated by sampling parameters from the fit covariance, not by bootstrapping.

— Employment baselines pulled live from the World Bank Open Data API.

— Sector risks as truncated normals calibrated to McKinsey, WEF and OECD, with a hierarchical Bayesian PyMC model (non-centred parameterisation, partial pooling) as a drop-in alternative.

— Vectorised Monte Carlo over (samples, years, sectors). n = 5,000 over 26 years and 11 sectors completes in roughly 26 ms; the naive triple-nested loop took ~30 s.

— Every fit and every run logged with seed, data version and serialised FitResult. Reproducibility before novelty.

Four named scenarios, peak displacement with full uncertainty propagation (95 percent Monte Carlo interval, n = 5000):

Optimistic, managed transition: peak 12.8 percent in 2027 — 6.7 to 10.0 percent at 2035
Base, no intervention: peak 25.4 percent in 2029 — 21.4 to 27.6 percent at 2035
Pessimistic, structural collapse: peak 26.4 percent in 2030 — 22.9 to 29.3 percent at 2035
Mythos-accelerated, cybersecurity cascade: peak 26.3 percent in 2031 — 22.2 to 28.7 percent at 2035

A note the headline misses: under the fitted SWE-bench curve the high-displacement scenarios converge, because the AI capability sigmoid saturates around 2030 regardless of which intervention path you assume. The real spread sits in the sector concentration, not the global peak — Administrative Services hits 57 percent by 2040.

Site: https://ai-labor-impact.vercel.app
Repo: https://github.com/dragoscalin33/ai-labor-impact

## Synthesis notes

Picked finding-first::hookA as the winner because it leads with the most concrete falsifiable claim in the entire candidate set — a precise number (82.4% predicted, 93.9% actual) that survives LinkedIn's 200-char truncation and forces the reader to keep scrolling to understand how the prediction was made. The tightened variant folds the inflection year into the R-squared bullet, sharpens the Monte Carlo bullet to name the NumPy broadcast explicitly, and adds the non-centred parameterisation detail to the PyMC bullet — all signals of ML engineering craft without the methodology-essay framing of the other angles. Runner-up is methodology-first::hookB because it tests a different opening (a discipline statement, not a finding) and so will A/B against the winner on a meaningfully different audience reaction. All numbers cross-checked against web/public/data/headline.json (lines 19-20 for R-squared and inflection, 41-42 for base peak, 66-68 for Mythos delta, 70-76 for sector snapshot, 77-82 for speedup) and web/public/data/validation.json (lines 203-214 for the 8-point leave-last-out fit predicting 82.4% with Mythos inside the 95% CI). Both drafts honour the no-emoji, no-job-seeking, no-invented-statistics rules.
