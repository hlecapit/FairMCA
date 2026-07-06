# Fair Representation Evaluation

A small scikit-learn-compatible toolkit for evaluating fairness in unsupervised representation learning.

## Metrics

- **Percentage of variance retained**: cross-validated linear reconstruction score expressed as a percentage.
- **MMD**: maximum mean discrepancy between sensitive groups in the learned embedding.
- **Adversarial leakage**: cross-validated classifier performance for predicting the sensitive attribute from the embedding.

## Main API

- `UnsupervisedFairnessEvaluator`
- `percentage_variance_retained`
- `groupwise_mmd`
- `adversarial_leakage_score`

## Example

```python
from sklearn.decomposition import PCA
from fair_repr_eval import UnsupervisedFairnessEvaluator

model = UnsupervisedFairnessEvaluator(PCA(n_components=2), variance_cv=5, adversarial_cv=5)
model.fit(X, sensitive=gender)
print(model.metrics_)
```
