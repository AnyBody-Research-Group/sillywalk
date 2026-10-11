# PCAPredictor reference

Import the public class from the package:

```python
from sillywalk import PCAPredictor
```

This reference describes the source implementation. See
[Methods and limitations](../methods.md) for its mathematical assumptions.

## Constructor

```python
PCAPredictor(
    data,
    n_components=0.99,
    svd_solver="auto",
    variance_threshold=1e-8,
    relative_variance_ratio=1e-3,
    transformer=None,
)
```

| Argument                  | Meaning                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `data`                    | Eager dataframe supported by Narwhals, with named columns. pandas and Polars are covered by repository tests.                         |
| `n_components`            | Retained count (integer), explained-variance setting (float between 0 and 1) or `None`, passed to scikit-learn PCA.                   |
| `svd_solver`              | scikit-learn PCA solver. Valid solver/component combinations follow scikit-learn requirements.                                        |
| `variance_threshold`      | Minimum original-column sample variance. Default `1e-8`.                                                                              |
| `relative_variance_ratio` | Minimum `abs(std/(mean+1e-12))`. Default `1e-3`.                                                                                      |
| `transformer`             | Fittable per-feature, shape-preserving, invertible transformer. Default is `StandardScaler`. Non-centered transforms require caution. |

The constructor fits immediately. Selected numeric features must contain
finite, complete training data. Metadata is not retained as original strings.
See [Data conventions](../data-conventions.md) and
[Transformer requirements](../preprocessing.md#custom-transformer-requirements).

The compatibility probe does not make every passing transformer safe.
Prediction and component conversion omit the transformed PCA centering offset.
Use centered preprocessing. A generic `MinMaxScaler` is not generally reliable
for reconstruction, even when its shape and feature independence pass checks.

## `predict`

```python
model.predict(constraints=None, target_pcs=None)
```

a dictionary with every original column name and a predicted value.
Selected PCA columns are reconstructed and excluded columns retain their
recorded training means.

`constraints` is a mapping or an intended one-row dataframe of selected
feature values in original units. Unknown columns and excluded columns raise
`ValueError`. The dataframe conversion takes the first row, so do not use
multirow frames as batch input. Loop over individual cases instead.

`target_pcs` is an optional NumPy array with one value per retained component.
It biases the solution toward those coordinates. Use the model's
`pca_n_components`, not the number of selected features, for its length.

`None` returns column means and emits a warning without a component-space solve.
An empty mapping emits a warning and, for a model with selected features,
constrains the first selected feature to its mean before solving. With nonlinear
preprocessing, that need not match `predict(None)` exactly.

The most recent solved component vector is stored in `model.y_opt`.
`predict(None)` does not update it. The constraint solver can return approximate
results for generally infeasible systems. Verify residuals yourself.

```python
prediction = model.predict({"Height": 1.76})
```

## `parameters_to_components`

```python
model.parameters_to_components(parameters)
```

Accepts a mapping or one-row dataframe containing all `model.pca_columns`.
Extra columns are ignored. Missing selected parameters raise `ValueError`.
Returns a Python list of length `model.pca_n_components`.

```python
coordinates = model.parameters_to_components(prediction)
```

## `components_to_parameters`

```python
model.components_to_parameters(principal_components)
```

Accepts a one-dimensional numeric sequence or NumPy array with one entry per
retained component. Wrong length raises `ValueError`. Returns a dictionary of
all original columns, filling excluded columns with means.

```python
reconstructed = model.components_to_parameters(coordinates)
```

Truncated PCA is a projection. A round trip of an arbitrary training observation
is not guaranteed to recover the original values.

## `export_pca_data`

```python
model.export_pca_data(filename=None)
```

With a string or path-like filename, writes a compressed NumPy `.npz` archive
and returns `None`. Create the parent directory yourself. With `None`, returns
a rewound in-memory binary buffer.

The archive stores `means`, `stds`, `columns`, `pca_columns`,
`pca_eigenvectors`, `pca_eigenvalues`, `pca_explained_variance_ratio` and
serialized fitted `transformer` bytes.

```python
from pathlib import Path

model.export_pca_data(Path("model.npz"))
buffer = model.export_pca_data()
```

The fitted transformer is pickled. Treat model archives as trusted executable
artifacts and record the Python and dependency versions used to create them.
They are not a stable, safe format for exchanging models from untrusted sources.

## `from_pca_data`

```python
PCAPredictor.from_pca_data(
    filename=None,
    means=None,
    stds=None,
    columns=None,
    pca_columns=None,
    pca_eigenvectors=None,
    pca_eigenvalues=None,
    pca_explained_variance_ratio=None,
    transformer=None,
)
```

The usual path loads a trusted file produced by `export_pca_data`:

```python
loaded = PCAPredictor.from_pca_data("model.npz")
```

Loading uses NumPy with `allow_pickle=True` and explicitly unpickles the
transformer. Never load untrusted files. A saved archive without a transformer
entry raises `ValueError` rather than substituting a new scaler.

Alternatively, supply the full model data explicitly:

```python
copy = PCAPredictor.from_pca_data(
    means=model.means,
    stds=model.stds,
    columns=model.columns,
    pca_columns=model.pca_columns,
    pca_eigenvectors=model.pca_eigenvectors,
    pca_eigenvalues=model.pca_eigenvalues,
    pca_explained_variance_ratio=model.pca_explained_variance_ratio,
    transformer=model.transformer,
)
```

This route does not fit, revalidate preprocessing, or verify the consistency
of arbitrary supplied arrays. Use a known coherent fitted state.

## Model attributes

| Attribute                      | Shape or meaning                                      |
| ------------------------------ | ----------------------------------------------------- |
| `columns`                      | List of all original feature names, in input order    |
| `pca_columns`                  | List of selected feature names, in selected order     |
| `pca_low_variance_columns`     | Set of excluded names, including nonnumeric features  |
| `pca_n_components`             | Number of retained components                         |
| `means`, `stds`                | Original-column statistics, length `len(columns)`     |
| `pca_eigenvectors`             | Array shaped `(n_components, len(pca_columns))`       |
| `pca_eigenvalues`              | Variance per retained component                       |
| `pca_explained_variance_ratio` | Retained shares of total transformed-feature variance |
| `transformer`                  | Fitted transformer used for selected features         |
| `y_opt`                        | Latest solved component vector, initially `None`      |

## Errors and warnings to investigate

Invalid dataframe values or PCA settings may raise backend/scikit-learn errors.
Missing transformer methods raise `TypeError`. Unsupported transformer shape or
feature mixing raises `ValueError`. Inconsistent parallel constraints raise
`ValueError`. Tiny retained eigenvalues emit a warning and are floored during
the KKT solve. These checks do not cover every modeling failure.
