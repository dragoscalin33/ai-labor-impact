"""
Metaculus Forecasting API Client
==================================
Fetches crowd-aggregated probability forecasts for AGI-related questions
from Metaculus. No authentication required for public questions.

API docs: https://www.metaculus.com/api/
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[2] / "data" / "cache"
BASE_URL = "https://www.metaculus.com/api2"

# Curated AGI-related question IDs on Metaculus
# Verified as of early 2026 — titles included for transparency
AGI_QUESTIONS: dict[str, int] = {
    "agi_before_2030": 5121,        # "Will there be a general AI by 2030?"
    "agi_by_2040": 3479,            # "Transformative AI by 2040?"
    "agi_2026_date": 3479,          # Community date estimate
    "superforecaster_agi": 11861,   # Superforecaster AGI timeline
    "weak_agi_date": 6099,          # Date of first weakly general AI
    "asi_before_2100": 4010,        # "Will AGI lead to ASI within a century?"
}


class MetaculusClient:
    """
    Lightweight client for Metaculus public forecasting API.

    Example
    -------
    >>> client = MetaculusClient()
    >>> df = client.get_question_history(5121)
    >>> summary = client.get_agi_forecast_summary()
    """

    def __init__(self, cache: bool = True) -> None:
        self.cache = cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ai-labor-impact/0.1 research"})

    def _cache_path(self, question_id: int) -> Path:
        return CACHE_DIR / f"metaculus_q{question_id}.parquet"

    def get_question_metadata(self, question_id: int) -> dict:
        """Fetch question metadata including current community prediction."""
        url = f"{BASE_URL}/questions/{question_id}/"
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def get_question_history(self, question_id: int) -> pd.DataFrame:
        """
        Fetch the full forecast history (community median over time) for a question.

        Returns
        -------
        pd.DataFrame with columns: timestamp, community_q1, community_median, community_q3
        """
        cache_path = self._cache_path(question_id)
        if self.cache and cache_path.exists():
            logger.info(f"Cache hit: metaculus q{question_id}")
            return pd.read_parquet(cache_path)

        url = f"{BASE_URL}/questions/{question_id}/"
        resp = self.session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        records = []
        history = data.get("community_prediction", {}).get("history", [])
        for point in history:
            records.append({
                "timestamp": pd.to_datetime(point["t"], unit="s", utc=True),
                "community_median": point.get("x2"),
                "community_q1": point.get("x1"),
                "community_q3": point.get("x3"),
                "question_id": question_id,
                "question_title": data.get("title", ""),
            })

        df = pd.DataFrame(records)

        if self.cache and not df.empty:
            df.to_parquet(cache_path, index=False)

        return df

    def get_agi_forecast_summary(self) -> pd.DataFrame:
        """
        Retrieve current community predictions for all tracked AGI questions.

        Returns a summary DataFrame suitable for plotting.
        """
        records = []
        for name, qid in AGI_QUESTIONS.items():
            try:
                meta = self.get_question_metadata(qid)
                cp = meta.get("community_prediction", {})
                latest = cp.get("full", {})

                records.append({
                    "question_key": name,
                    "question_id": qid,
                    "title": meta.get("title", name),
                    "community_median": latest.get("q2"),
                    "community_q1": latest.get("q1"),
                    "community_q3": latest.get("q3"),
                    "num_predictions": meta.get("number_of_predictions", None),
                    "resolved": meta.get("resolution") is not None,
                    "url": f"https://www.metaculus.com/questions/{qid}/",
                })
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"Skipping question {qid} ({name}): {e}")

        return pd.DataFrame(records)
