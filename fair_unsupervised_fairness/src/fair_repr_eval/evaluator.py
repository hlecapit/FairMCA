from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_array, check_is_fitted

from .metrics import adversarial_leakage_score, component_sensitive_corr, groupwise_mmd, percentage_variance_retained


class UnsupervisedFairnessEvaluator(BaseEstimator):
    """Scikit-learn compatible evaluator for unsupervised representation learning.

    Parameters
    ----------
    estimator:
        Unsupervised model to evaluate. It should expose ``fit`` and ideally
        ``transform`` or ``fit_transform``. If neither is available, the fitted
        estimator may expose an ``embedding_`` attribute.
    variance_cv:
        Number of CV splits used for the variance-retention decoder.
    mmd_gamma:
        Optional RBF kernel width for MMD. If ``None``, a median heuristic is used.
    adversarial_classifier:
        Optional classifier used to predict the sensitive attribute from the embedding.
    adversarial_cv:
        Number of CV splits for the adversarial classifier.
    random_state:
        Random seed for cross-validation splitting.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        variance_cv: int = 5,
        mmd_gamma: float | None = None,
        adversarial_classifier: Any | None = None,
        adversarial_cv: int = 5,
        random_state: int = 0,
    ) -> None:
        self.estimator = estimator
        self.variance_cv = variance_cv
        self.mmd_gamma = mmd_gamma
        self.adversarial_classifier = adversarial_classifier
        self.adversarial_cv = adversarial_cv
        self.random_state = random_state

    def _embedding_from_fitted(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator_, "transform"):
            return self.estimator_.transform(X)
        if hasattr(self.estimator_, "embedding_"):
            return getattr(self.estimator_, "embedding_")
        raise AttributeError(
            "The wrapped estimator must provide transform, fit_transform, or embedding_."
        )

    def fit(self, X: Any, y: Any = None, sensitive: Any | None = None):
        if sensitive is None:
            raise ValueError("sensitive must be provided to fit the fairness evaluator.")

        X = check_array(X, ensure_2d=True, dtype=float)
        self.estimator_ = clone(self.estimator)
        if hasattr(self.estimator_, "fit_transform"):
            embedding = self.estimator_.fit_transform(X, y)
        else:
            self.estimator_.fit(X, y)
            embedding = self._embedding_from_fitted(X)

        self.X_fit_ = X
        self.sensitive_ = np.asarray(sensitive)
        self.embedding_ = check_array(embedding, ensure_2d=True, dtype=float)
        self.metrics_ = self.evaluate(X=X, sensitive=sensitive)

        self.variance_retained_ = self.metrics_["variance_retained_percent"]
        self.mmd_ = self.metrics_["mmd_mean"]
        self.adversarial_ = {
            "accuracy_mean": self.metrics_["adversarial_accuracy_mean"],
            "balanced_accuracy_mean": self.metrics_["adversarial_balanced_accuracy_mean"],
        }
        return self

    def transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes=["estimator_"])
        X = check_array(X, ensure_2d=True, dtype=float)
        if hasattr(self.estimator_, "transform"):
            return check_array(self.estimator_.transform(X), ensure_2d=True, dtype=float)
        raise AttributeError("The wrapped estimator does not implement transform().")

    def evaluate(self, X: Any | None = None, sensitive: Any | None = None) -> dict[str, Any]:
        check_is_fitted(self, attributes=["estimator_", "embedding_"])

        if X is None:
            X = self.X_fit_
            embedding = self.embedding_
        else:
            X = check_array(X, ensure_2d=True, dtype=float)
            embedding = self.transform(X)

        if sensitive is None:
            sensitive = self.sensitive_

        variance_retained = percentage_variance_retained(
            X,
            embedding,
            cv=self.variance_cv,
            random_state=self.random_state,
        )
        mmd_result = groupwise_mmd(embedding, sensitive, gamma=self.mmd_gamma)
        adversarial_result = adversarial_leakage_score(
            embedding,
            sensitive,
            classifier=self.adversarial_classifier,
            cv=self.adversarial_cv,
            random_state=self.random_state,
        )
        corr_result = component_sensitive_corr(embedding, sensitive)

        return {
            "variance_retained_percent": variance_retained,
            "mmd_mean": mmd_result["mean"],
            "mmd_max": mmd_result["max"],
            "mmd_min": mmd_result["min"],
            "mmd_gamma": mmd_result["gamma"],
            "mmd_pairwise": mmd_result["pairwise"],
            "adversarial_accuracy_mean": adversarial_result["accuracy_mean"],
            "adversarial_accuracy_std": adversarial_result["accuracy_std"],
            "adversarial_balanced_accuracy_mean": adversarial_result["balanced_accuracy_mean"],
            "adversarial_balanced_accuracy_std": adversarial_result["balanced_accuracy_std"],
            "adversarial_majority_class_baseline": adversarial_result["majority_class_baseline"],
            "adversarial_delta_accuracy": adversarial_result["delta_accuracy"],
            "corr_components": corr_result.tolist(),
        }
