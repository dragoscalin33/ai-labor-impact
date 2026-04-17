# LinkedIn launch — copy / paste templates

Three drafts in descending length so you can pick the format that fits
your audience. **Replace** every `[…]` placeholder before posting.

---

## 1 · Long-form post  (recommended for the launch)

> 🤖 **What does Claude Mythos Preview's 93.9 % SWE-bench actually mean for jobs?**
>
> I built a fully-reproducible, fully-Bayesian observatory to find out.
> Live demo (no API key, no install): [https://ai-labor-impact.streamlit.app]
>
> The TL;DR — projecting global unemployment under four scenarios with Monte
> Carlo n = 5,000:
>
> 🟢 Optimistic (managed transition): peak **12-18 %**
> 🟠 Base case (no major intervention): peak **28-38 %**
> 🔴 Pessimistic (structural collapse): peak **45-58 %**
> 🟣 Mythos-Accelerated (cybersecurity cascade): peak **35-52 %**, crisis dynamics from 2027
>
> What makes this different from the usual AI-vs-jobs takes:
>
> 📈 Sigmoid is **fitted** to real SWE-bench scores, not assumed
> 🎲 5,000 Monte Carlo draws, fully **vectorised** in NumPy (30 s → 26 ms, > 1000× faster than the naive loop)
> 🧠 Hierarchical Bayesian sector-risk model in PyMC with non-centred parameterisation
> 🔍 **Leave-last-out temporal CV** — the model fitted on data BEFORE April 2026 placed Mythos inside its 95 % CI. Predictive, not descriptive.
> 📦 Every run is MLflow-tracked, every fit is pickled with its seed and data version. Auditable end-to-end.
>
> Built with: Python · NumPy · SciPy · PyMC · MLflow · Streamlit · Plotly · Docker · GitHub Actions
>
> The repo is a deliberate ML-engineering portfolio piece — code is type-checked, ruff-clean, tested on Python 3.10/3.11/3.12.
>
> 👇 If you're hiring for ML / ML-Eng roles, I'd love to chat. DM open.
>
> 🔗 Demo: [https://ai-labor-impact.streamlit.app]
> 🔗 Repo: [https://github.com/dragoscalin33/ai-labor-impact]
> 🔗 Methodology notebooks: in repo, especially `notebooks/05_temporal_validation.ipynb`
>
> #MachineLearning #MLEngineering #DataScience #PyMC #BayesianStatistics #AIimpact #LaborMarket #AIsafety

---

## 2 · Mid-length  (for your headline + cross-post)

> Spent the weekend turning *"how do I quantify Claude Mythos Preview's labor-market impact"* into a real model.
>
> 🟢 Sigmoid fit to real SWE-bench scores
> 🎲 5,000-sample Monte Carlo, vectorised in NumPy (~26 ms for the whole pipeline)
> 🧠 Hierarchical Bayesian sector-risk model in PyMC
> 🔍 Time-series CV: the model trained on pre-April-2026 data placed Mythos inside its 95 % CI
> 📊 Live, no-key Streamlit demo: [https://ai-labor-impact.streamlit.app]
>
> Open to ML / ML-Eng roles.
>
> Repo: [https://github.com/dragoscalin33/ai-labor-impact]
> #MachineLearning #PyMC #DataScience

---

## 3 · One-liner  (for Twitter / Bluesky cross-post)

> Built a fully-Bayesian observatory of AI's labor-market impact. Live demo (no key): [https://ai-labor-impact.streamlit.app] · Repo: [https://github.com/dragoscalin33/ai-labor-impact] · #MLEngineering

---

## Posting checklist

Before hitting publish:

- [ ] Replace `[https://ai-labor-impact.streamlit.app]` with the *real* Streamlit URL after deploying.
- [ ] Replace the GitHub URL if your repo path differs.
- [ ] Add a screenshot or short screen-capture GIF of the dashboard (tab "📉 Unemployment Scenarios" is the most striking).
- [ ] Pin the post to your profile.
- [ ] Set your LinkedIn headline to: *"ML Engineer · Building reproducible, Bayesian, end-to-end ML pipelines · Open to roles"* (or similar).
- [ ] Update your "Featured" section with the demo + repo links.
- [ ] When recruiters comment, reply within a few hours — first 12 h drives most of the reach.

## Asset suggestions

Screenshots that read well as the post's lead image:

1. **Tab 2 — Unemployment Scenarios** fan chart (the headline visual).
2. **Tab 1 — Capability curve** with the Mythos star highlighted.
3. **Notebook 05** showing the temporal-CV scatter where Mythos lands inside the 95 % CI — strongest narrative for ML reviewers.

If you record a short GIF (Loom / Kap / `gifsicle`), focus on the slider
movement (`n samples`) and switching between scenarios — that *visually*
communicates "real Monte Carlo + interactive" in 6 seconds.
