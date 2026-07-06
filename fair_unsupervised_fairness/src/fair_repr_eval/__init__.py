"""Fairness evaluation tools for unsupervised representation learning."""

from .evaluator import UnsupervisedFairnessEvaluator
from .metrics import adversarial_leakage_score, groupwise_mmd, percentage_variance_retained

__all__ = [
    "UnsupervisedFairnessEvaluator",
    "adversarial_leakage_score",
    "groupwise_mmd",
    "percentage_variance_retained",
]

__version__ = "0.1.0"
