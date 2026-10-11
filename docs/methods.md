# Methods and limitations

## Fitting

The constructor computes column means, sample standard deviations and sample
variances on the original named dataframe. Nonnumeric columns are replaced
internally by NaN. A feature is selected when both conditions hold:

```text
variance >= variance_threshold
abs(std / (mean + 1e-12)) >= relative_variance_ratio
```

Defaults are `variance_threshold=1e-8` and `relative_variance_ratio=1e-3`.
The first threshold depends on the original units. The second is a heuristic
that behaves differently for means near zero. Screening happens before
transformation. Inspect `pca_columns` and `pca_low_variance_columns` rather
than assuming every numeric column is retained.

The selected matrix is transformed, using `StandardScaler` by default, then
passed to `sklearn.decomposition.PCA`. `n_components=0.99` is the default
explained-variance retention setting. An integer selects a component count.
`pca_explained_variance_ratio` records each retained component's share of the
total variance in transformed feature space, not predictive accuracy.

There is no iterative missing-data likelihood fit. The historical ML-PCA
description refers to the constrained statistical reconstruction context and
should not be read as an implementation of incomplete-data ML-PCA or
probabilistic PCA.

## Constrained prediction

Let `V` contain the retained PCA eigenvectors as rows and `y` the component
coordinates. Under centered preprocessing, a transformed feature vector is
reconstructed as `V.T @ y`. A partial constraint becomes a row of a linear
system `B @ y = d`.

Prediction seeks a solution close to `target_pcs`, using inverse eigenvalues
as the penalty weights:

```text
minimize 0.5 * (y - target).T @ diag(1/eigenvalues) @ (y - target)
subject to B @ y = d
```

The default target is the zero component vector. This is a Gaussian-prior
mode/conditional reconstruction interpretation in the retained component
space, not a fitted causal model or a physical optimality criterion.

The implementation solves the KKT system using `numpy.linalg.lstsq`,
reconstructs selected features, then inverse-transforms them. Original
columns excluded from PCA are filled with their training means.

Consistent parallel constraints are dropped as redundant. Inconsistent
parallel pairs raise `ValueError`. The code does not detect every generally
infeasible constraint system, nor does it verify all output residuals. A
least-squares result can therefore be approximate.

Always check requested values explicitly:

```python
import numpy as np

constraints = {"Height": 1.76}
prediction = model.predict(constraints)
for name, requested in constraints.items():
    if not np.isclose(prediction[name], requested, rtol=0, atol=1e-6):
        raise ValueError(f"Constraint residual exceeds tolerance for {name}")
```

Choose tolerances in each variable's units. Retaining more components can
make constraints representable, but does not by itself establish a better
model. Numerically tiny eigenvalues are floored during prediction with a
warning. Treat that warning as a reason to examine redundant features and
conditioning, not as evidence the data problem was solved.

## Component conversion

`parameters_to_components` transforms a complete set of selected feature
values and projects it into the retained component space.
`components_to_parameters` reconstructs the selected features and fills
excluded columns with means. Truncation means a parameter-to-component
round trip need not reproduce the original observation.

The implementation does not store scikit-learn's transformed `pca.mean_`
offset. Default `StandardScaler` centers features, so that offset is
numerically near zero. Arbitrary non-centered custom transformers are not
generally safe for prediction or component conversion. See
[Preprocessing and transformers](preprocessing.md).

## Limitations

- Selected training features must be complete and finite. The estimator does
  not impute missing measurements or account for feature-specific measurement
  uncertainty.
- PCA models linear structure in transformed feature space. It does not enforce
  anatomical bounds, joint limits, positive scalars, category membership,
  contact consistency or dynamic feasibility.
- Constraint systems can be infeasible after truncation. General feasibility,
  output residual tolerances and physically plausible predictions require
  caller checks.
- Numeric identifiers and category codes are treated as continuous data.
  Nonnumeric metadata is not preserved in predicted outputs.
- Rows do not carry independence assumptions. Multiple cycles from one subject
  must not be counted automatically as multiple independent subjects.
- There are no confidence intervals, out-of-distribution warnings or validated
  clinical interpretations.
- Fourier output assumes uniformly spaced, phase-aligned periodic samples.
  Endpoint correction changes data and does not establish physical periodicity.
- Rendered AnyScript is text, not a successfully loaded or simulated model.
  A compatible AnyBody/AMMR installation and model-specific validation are
  required.
- Serialized `.npz` models contain pickle data. Load trusted artifacts only and
  retain the software environment needed to deserialize the transformer.

If no features pass screening, the model has zero components.
`predict(None)` returns the recorded means with a warning. Other prediction
or conversion paths are not a useful general zero-feature workflow. Check
`pca_n_components > 0` before using them.

## Validation for research use

Define a validation plan before interpreting predictions. Record the training
population and provenance, split observations appropriately, fit preprocessing
only on training data, and compare predictions with held-out observations.
Measure errors in meaningful units and compare against suitable baselines.
Separate software tests, statistical validation and physical simulation
verification. Passing the synthetic example establishes only the software
behaviors checked by that example.
