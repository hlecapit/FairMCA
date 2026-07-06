from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


def _ensure_2d(array: Any) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return check_array(data, ensure_2d=True, dtype=float)


def _safe_n_splits(y: np.ndarray, requested_splits: int) -> int:
    unique, counts = np.unique(y, return_counts=True)
    if unique.size < 2:
        raise ValueError("Sensitive attribute must contain at least two groups.")
    return max(2, min(int(requested_splits), int(counts.min()), int(len(y))))


def percentage_variance_retained(
    X: Any,
    embedding: Any,
    *,
    cv: int = 5,
    ridge_alpha: float = 1e-6,
    random_state: int = 0,
) -> float:
    """Estimate the percentage of variance retained by a learned embedding.

    The metric standardizes the original data, fits a linear decoder from the
    embedding to the standardized input on each fold, and returns the cross-
    validated percentage of explained variance.
    """
    X = _ensure_2d(X)
    embedding = _ensure_2d(embedding)
    if X.shape[0] != embedding.shape[0]:
        raise ValueError("X and embedding must have the same number of rows.")

    Xs = StandardScaler().fit_transform(X)
    n_splits = max(2, min(int(cv), X.shape[0]))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    sse = 0.0
    sst = 0.0
    for train_idx, test_idx in splitter.split(Xs):
        decoder = Ridge(alpha=ridge_alpha)
        decoder.fit(embedding[train_idx], Xs[train_idx])
        reconstruction = decoder.predict(embedding[test_idx])
        residual = Xs[test_idx] - reconstruction
        sse += float(np.sum(residual**2))
        centered = Xs[test_idx] - Xs[test_idx].mean(axis=0, keepdims=True)
        sst += float(np.sum(centered**2))

    if sst <= 0:
        return 100.0
    score = max(0.0, 1.0 - sse / sst)
    return float(100.0 * score)


def _median_heuristic_gamma(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 1.0
    diffs = X[:, None, :] - X[None, :, :]
    sq_dists = np.sum(diffs * diffs, axis=-1)
    tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    median_sq_dist = float(np.median(tri))
    if median_sq_dist <= 0:
        return 1.0
    return 1.0 / (2.0 * median_sq_dist)


def _unbiased_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    Kxx = pairwise_kernels(X, X, metric="rbf", gamma=gamma)
    Kyy = pairwise_kernels(Y, Y, metric="rbf", gamma=gamma)
    Kxy = pairwise_kernels(X, Y, metric="rbf", gamma=gamma)

    m = X.shape[0]
    n = Y.shape[0]
    if m > 1:
        term_xx = (np.sum(Kxx) - np.trace(Kxx)) / (m * (m - 1))
    else:
        term_xx = 0.0
    if n > 1:
        term_yy = (np.sum(Kyy) - np.trace(Kyy)) / (n * (n - 1))
    else:
        term_yy = 0.0
    term_xy = 2.0 * float(np.mean(Kxy))
    return float(term_xx + term_yy - term_xy)


def groupwise_mmd(
    embedding: Any,
    sensitive: Any,
    *,
    gamma: float | None = None,
    unbiased: bool = True,
) -> dict[str, Any]:
    """Compute pairwise MMD between sensitive groups in an embedding.

    Returns the pairwise values, their mean, max, and min.
    """
    embedding = _ensure_2d(embedding)
    sensitive = np.asarray(sensitive)
    if embedding.shape[0] != sensitive.shape[0]:
        raise ValueError("embedding and sensitive must have the same number of rows.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(sensitive)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Sensitive attribute must contain at least two groups.")

    if gamma is None:
        gamma = _median_heuristic_gamma(embedding)

    pairwise_scores: dict[str, float] = {}
    for a, b in combinations(unique_labels, 2):
        Xa = embedding[labels == a]
        Xb = embedding[labels == b]
        if Xa.shape[0] == 0 or Xb.shape[0] == 0:
            continue
        if unbiased:
            value = _unbiased_mmd2(Xa, Xb, gamma)
        else:
            Kaa = pairwise_kernels(Xa, Xa, metric="rbf", gamma=gamma)
            Kbb = pairwise_kernels(Xb, Xb, metric="rbf", gamma=gamma)
            Kab = pairwise_kernels(Xa, Xb, metric="rbf", gamma=gamma)
            value = float(np.mean(Kaa) + np.mean(Kbb) - 2.0 * np.mean(Kab))
        pairwise_scores[f"{encoder.inverse_transform([a])[0]}__vs__{encoder.inverse_transform([b])[0]}"] = float(max(value, 0.0))

    values = np.array(list(pairwise_scores.values()), dtype=float)
    if values.size == 0:
        raise ValueError("Unable to compute pairwise MMD values.")

    return {
        "pairwise": pairwise_scores,
        "mean": float(values.mean()),
        "max": float(values.max()),
        "min": float(values.min()),
        "gamma": float(gamma),
    }


def adversarial_leakage_score(
    embedding: Any,
    sensitive: Any,
    *,
    classifier: Any | None = None,
    cv: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    """Predict sensitive labels from the embedding with a cross-validated classifier."""
    embedding = _ensure_2d(embedding)
    sensitive = np.asarray(sensitive)
    if embedding.shape[0] != sensitive.shape[0]:
        raise ValueError("embedding and sensitive must have the same number of rows.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(sensitive)
    n_splits = _safe_n_splits(y, cv)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if classifier is None:
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs")

    accuracy = cross_val_score(classifier, embedding, y, cv=splitter, scoring="accuracy")
    balanced_accuracy = cross_val_score(classifier, embedding, y, cv=splitter, scoring="balanced_accuracy")

    return {
        "accuracy_mean": float(np.mean(accuracy)),
        "accuracy_std": float(np.std(accuracy)),
        "balanced_accuracy_mean": float(np.mean(balanced_accuracy)),
        "balanced_accuracy_std": float(np.std(balanced_accuracy)),
        "majority_class_baseline": float(np.max(np.bincount(y) / y.size)),
        "delta_accuracy": float(np.mean(accuracy) - np.max(np.bincount(y) / y.size)),
    }


def component_sensitive_corr(embedding: Any, sensitive: Any) -> np.ndarray:
    """Compute the correlation between each embedding component and the sensitive attribute."""
    embedding = _ensure_2d(embedding)
    sensitive = np.asarray(sensitive)
    if embedding.shape[0] != sensitive.shape[0]:
        raise ValueError("embedding and sensitive must have the same number of rows.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(sensitive)
    y_centered = y - np.mean(y)

    correlations = []
    for i in range(embedding.shape[1]):
        x = embedding[:, i]
        x_centered = x - np.mean(x)
        numerator = float(np.sum(x_centered * y_centered))
        denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
        if denominator == 0:
            corr = 0.0
        else:
            corr = numerator / denominator
        correlations.append(corr)

    return np.array(correlations, dtype=float)