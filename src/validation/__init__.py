"""
Model validation utilities.

This package contains rigorous out-of-sample validation tools that test
whether the AI capability sigmoid is honestly predictive (vs. merely
fitting the data it was given).
"""

from .temporal_cv import (
    TemporalCVResult,
    leave_last_out,
    rolling_origin_cv,
)

__all__ = ["TemporalCVResult", "leave_last_out", "rolling_origin_cv"]
