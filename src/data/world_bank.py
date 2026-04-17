"""
World Bank API Client
======================
Fetches real employment and unemployment data from the World Bank
Open Data API (no authentication required).

Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.worldbank.org/v2"
CACHE_DIR = Path(__file__).parents[3] / "data" / "cache"

# Key indicators
INDICATORS: dict[str, str] = {
    "SL.UEM.TOTL.ZS": "unemployment_rate_pct",          # Unemployment, total (% labor force)
    "SL.UEM.TOTL.NE.ZS": "unemployment_ne_pct",         # Unemployment, ILO modeled (%)
    "SL.TLF.TOTL.IN": "labor_force_total",               # Labor force, total
    "SL.EMP.TOTL.SP.ZS": "employment_rate_pct",          # Employment to population ratio
    "SL.IND.EMPL.ZS": "employment_industry_pct",         # Employment in industry (%)
    "SL.SRV.EMPL.ZS": "employment_services_pct",         # Employment in services (%)
    "SL.AGR.EMPL.ZS": "employment_agriculture_pct",      # Employment in agriculture (%)
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",               # GDP growth (annual %)
}

# Country groups for aggregated analysis
REGIONS: dict[str, str] = {
    "WLD": "World",
    "HIC": "High Income",
    "MIC": "Middle Income",
    "LIC": "Low Income",
    "EUU": "European Union",
    "NAC": "North America",
    "EAS": "East Asia & Pacific",
    "LCN": "Latin America & Caribbean",
}


class WorldBankClient:
    """
    Client for World Bank Open Data API.

    Implements caching to avoid redundant API calls during development.

    Example
    -------
    >>> client = WorldBankClient()
    >>> df = client.get_indicator(
    ...     "SL.UEM.TOTL.ZS", countries=["WLD", "HIC"], date_range=(2000, 2024),
    ... )
    """

    def __init__(self, cache: bool = True, per_page: int = 1000) -> None:
        self.cache = cache
        self.per_page = per_page
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, indicator: str, countries: str, dates: str) -> Path:
        safe = f"{indicator}_{countries}_{dates}".replace("/", "-").replace(",", "_")
        return CACHE_DIR / f"wb_{safe}.parquet"

    def get_indicator(
        self,
        indicator: str,
        countries: list[str] | None = None,
        date_range: tuple[int, int] = (2000, 2024),
    ) -> pd.DataFrame:
        """
        Fetch a World Bank indicator time series.

        Parameters
        ----------
        indicator : str
            World Bank indicator code (e.g. 'SL.UEM.TOTL.ZS').
        countries : list[str], optional
            ISO3 country/region codes. Defaults to all World Bank regions.
        date_range : tuple[int, int]
            (start_year, end_year) inclusive.

        Returns
        -------
        pd.DataFrame with columns: country_code, country_name, year, value, indicator
        """
        countries = countries or list(REGIONS.keys())
        country_str = ";".join(countries)
        date_str = f"{date_range[0]}:{date_range[1]}"

        cache_path = self._cache_path(indicator, country_str[:30], date_str)
        if self.cache and cache_path.exists():
            logger.info(f"Cache hit: {cache_path.name}")
            return pd.read_parquet(cache_path)

        url = f"{BASE_URL}/country/{country_str}/indicator/{indicator}"
        params = {
            "format": "json",
            "per_page": self.per_page,
            "date": date_str,
            "mrv": 30,
        }

        logger.info(f"Fetching {indicator} for {countries} ({date_str})")
        all_records: list[dict] = []
        page = 1

        while True:
            params["page"] = page
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error(f"World Bank API error: {e}")
                raise

            if len(data) < 2 or not data[1]:
                break

            for row in data[1]:
                if row.get("value") is not None:
                    all_records.append({
                        "country_code": row["countryiso3code"],
                        "country_name": row["country"]["value"],
                        "year": int(row["date"]),
                        "value": float(row["value"]),
                        "indicator": indicator,
                        "indicator_name": INDICATORS.get(indicator, indicator),
                    })

            # Pagination
            total_pages = data[0].get("pages", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.1)  # polite rate limiting

        df = pd.DataFrame(all_records).sort_values(["country_code", "year"])

        if self.cache and not df.empty:
            df.to_parquet(cache_path, index=False)

        return df

    def get_employment_dashboard(
        self,
        date_range: tuple[int, int] = (2000, 2024),
    ) -> pd.DataFrame:
        """
        Fetch a multi-indicator employment overview for all major regions.

        Returns a wide DataFrame: one row per (country, year), one column per indicator.
        """
        dfs = []
        for indicator, col_name in INDICATORS.items():
            try:
                df = self.get_indicator(indicator, date_range=date_range)
                df = df.rename(columns={"value": col_name})[
                    ["country_code", "country_name", "year", col_name]
                ]
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {indicator}: {e}")

        if not dfs:
            raise RuntimeError("No data retrieved from World Bank API.")

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=["country_code", "country_name", "year"], how="outer")

        return merged.sort_values(["country_name", "year"]).reset_index(drop=True)

    def get_global_unemployment_trend(
        self,
        date_range: tuple[int, int] = (2000, 2024),
    ) -> pd.DataFrame:
        """
        Returns clean global unemployment trend data.
        Convenient for quick plotting.
        """
        df = self.get_indicator(
            "SL.UEM.TOTL.ZS",
            countries=list(REGIONS.keys()),
            date_range=date_range,
        )
        df["region_label"] = df["country_code"].map(REGIONS).fillna(df["country_name"])
        return df
