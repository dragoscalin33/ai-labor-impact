# LinkedIn launch — copy / paste templates

Two drafts. Pick one, replace `[…]` placeholders, post.

---

## 1 · Standard launch post

> Project release: an observatory of how AI capability progression projects onto global labour displacement. Reproducible, Bayesian, end-to-end.
>
> Live dashboard, no install, no API key required: [https://ai-labor-impact.streamlit.app]
> Repo: [https://github.com/dragoscalin33/ai-labor-impact]
>
> Methodology
>
> — Logistic curve fitted to published SWE-bench, HumanEval and MMLU scores by bounded non-linear least squares. Parameter uncertainty is propagated via Monte Carlo draws of the fit covariance.
>
> — Vectorised Monte Carlo over (samples × years × sectors). n = 5,000 over 26 years and 11 sectors completes in ~26 ms on a laptop; the equivalent triple-nested Python loop took ~30 s.
>
> — Hierarchical Bayesian sector-risk model in PyMC, non-centred parameterisation. Partial pooling regularises the consensus estimates from McKinsey, WEF and OECD.
>
> — Leave-last-out temporal cross-validation. The sigmoid fitted exclusively on SWE-bench data published before April 2026 placed the Claude Mythos Preview observation (93.9 %) within its 95 % confidence interval.
>
> — Every fit and every Monte Carlo run is logged to MLflow with seed, data version and serialised FitResult.
>
> Headline numbers, peak unemployment with full uncertainty propagation, four named scenarios:
>
> — Optimistic, managed transition: 12 – 18 %
> — Base, no major intervention: 28 – 38 %
> — Pessimistic, structural collapse: 45 – 58 %
> — Mythos-accelerated, cybersecurity cascade: 35 – 52 %, with crisis dynamics from 2027
>
> Stack: Python, NumPy, SciPy, PyMC, MLflow, Streamlit, Plotly, Docker, GitHub Actions. Type-checked, ruff-clean, tested on Python 3.10, 3.11 and 3.12.
>
> Open to conversations about ML engineering work — DMs are open.

---

## 2 · Short cross-post (Twitter / Bluesky / Mastodon)

> Released a reproducible Bayesian observatory of how AI capability progression maps onto global labour displacement. Vectorised Monte Carlo, hierarchical PyMC, temporal cross-validation. Live, no install: [https://ai-labor-impact.streamlit.app]. Repo: [https://github.com/dragoscalin33/ai-labor-impact].

---

## Posting checklist

- Replace the placeholder Streamlit URL once the deploy is live.
- Add a screenshot or short screen-capture as the post media. The Monte Carlo fan chart (Unemployment scenarios tab) reads cleanest.
- Pin the post to your profile.
- Update the LinkedIn headline to a single line: *Machine Learning Engineer · Reproducible, Bayesian, end-to-end ML pipelines.*
- Add the demo and repo links to the *Featured* section.
- Reply to comments within the first twelve hours. That window drives most of the reach on LinkedIn.

## Asset suggestions

Three screenshots worth taking:

1. The fan chart on the *Unemployment scenarios* tab. Strongest single image.
2. The *Capability curve* tab with the Mythos data point at the right edge of the sigmoid.
3. The leave-last-out scatter from `notebooks/05_temporal_validation.ipynb` showing Mythos inside the 95 % CI. Strongest narrative image for technical reviewers.

If you record a short screen capture, focus on moving the *Monte Carlo samples* slider and switching between scenarios. That communicates *real Monte Carlo, interactive* in roughly six seconds without any voice-over.
