"""
Profiling utilities for NSA optimizer ablation study.
"""
from .flops import (
    calculate_flops_per_token,
    calculate_arithmetic_intensity,
    format_flops,
    format_arithmetic_intensity,
)

__all__ = [
    "calculate_flops_per_token",
    "calculate_arithmetic_intensity",
    "format_flops",
    "format_arithmetic_intensity",
]
