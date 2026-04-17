"""Data ingestion modules for real-world APIs."""
from .ai_benchmarks import BENCHMARK_CATALOG, get_benchmark_dataframe
from .metaculus import MetaculusClient
from .world_bank import WorldBankClient

__all__ = ["WorldBankClient", "get_benchmark_dataframe", "BENCHMARK_CATALOG", "MetaculusClient"]
