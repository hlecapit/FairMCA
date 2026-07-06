import numpy as np
from sklearn.decomposition import PCA

from fair_repr_eval import UnsupervisedFairnessEvaluator


def test_evaluator_fit_and_metrics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 6))
    sensitive = np.array([0] * 30 + [1] * 30)
    model = UnsupervisedFairnessEvaluator(PCA(n_components=2, random_state=0), variance_cv=3, adversarial_cv=3)
    fitted = model.fit(X, sensitive=sensitive)

    assert fitted is model
    assert 0.0 <= model.variance_retained_ <= 100.0
    assert model.mmd_ >= 0.0
    assert 0.0 <= model.adversarial_["accuracy_mean"] <= 1.0
    assert 0.0 <= model.adversarial_["balanced_accuracy_mean"] <= 1.0
    assert "mmd_pairwise" in model.metrics_
