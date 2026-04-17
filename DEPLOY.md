# Deploying the Dashboard

Three supported paths. Pick whichever matches your audience.

| Path                         | Use for                              | Time | Cost |
|------------------------------|--------------------------------------|------|------|
| **Streamlit Community Cloud** | Public link to share on LinkedIn / CV | 5 min | Free |
| **Docker**                    | Reviewers who want full local control | 1 min | Free |
| **Local Python**              | You, while iterating                 | <1 min | Free |

---

## A) Streamlit Community Cloud  *(recommended for the LinkedIn post)*

1. **Push the repo to GitHub** (it should already be public):
   ```bash
   git push origin main
   ```
2. **Sign in** at <https://share.streamlit.io> with your GitHub account.
3. Click **“New app”** and fill in:
   - **Repository**: `dragoscalin33/ai-labor-impact`
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
   - **Python version**: 3.11 (under *Advanced settings*)
4. Click **Deploy**. First build takes ~3–5 min.
5. (Optional) **Secrets** — only needed if you want the *live* LLM Insights tab.
   In *App settings → Secrets* paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   The dashboard ships with `data/insights_cache.json` so this is **purely
   optional** — testers without a key still get rich narrative.
6. Copy the resulting URL (e.g. `https://ai-labor-impact.streamlit.app`)
   and paste it into the README's *Live demo* badge and into `LINKEDIN.md`.

> **Why the lean `requirements.txt`?** The Cloud install times out after
> ~10 min. Our default file omits PyMC, MLflow, and Jupyter so cold deploys
> finish in under 5 min. Bayesian and tracking notebooks remain runnable
> locally via `pip install -r requirements-dev.txt`.

---

## B) Docker

```bash
docker build -t ai-labor-impact .
docker run --rm -p 8501:8501 ai-labor-impact
# → open http://localhost:8501
```

Or with `docker-compose`:

```bash
docker compose up --build
```

The image is ~600 MB and contains *only* the lean runtime — perfect for
demoing on a reviewer's laptop without polluting their Python env.

---

## C) Local Python

```bash
git clone https://github.com/dragoscalin33/ai-labor-impact.git
cd ai-labor-impact
pip install -r requirements.txt
streamlit run app/dashboard.py
```

For the Bayesian and MLflow notebooks add:

```bash
pip install -r requirements-dev.txt
jupyter lab notebooks/
mlflow ui                       # opens http://localhost:5000
```
