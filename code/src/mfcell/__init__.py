"""Leakage-audited matrix-free residual propagation for gene measurements."""

from .experiment import run_experiment, verify_locked_results

__all__ = ["run_experiment", "verify_locked_results"]
__version__ = "2.0.0"
