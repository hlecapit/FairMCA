import numpy as np
from sklearn.decomposition import PCA

from fair_repr_eval.metrics import adversarial_leakage_score, groupwise_mmd, percentage_variance_retained


def test_percentage_variance_retained_returns_percentage():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    embedding = PCA(n_components=2, random_state=0).fit_transform(X)
    score = percentage_variance_retained(X, embedding, cv=4)
    assert 0.0 <= score <= 100.0


def test_groupwise_mmd_identical_groups_is_small():
    X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    sensitive = np.array([0, 0, 0, 1, 1, 1])
    result = groupwise_mmd(X, sensitive, unbiased=True)
    assert result["mean"] >= 0.0
    assert result["mean"] < 1e-8


def test_adversarial_leakage_score_is_high_on_separable_data():
    embedding = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]])
    sensitive = np.array([0, 0, 0, 1, 1, 1])
    result = adversarial_leakage_score(embedding, sensitive, cv=3, random_state=0)
    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert 0.0 <= result["balanced_accuracy_mean"] <= 1.0
    assert result["balanced_accuracy_mean"] > 0.9
