# Preprocessing and transformers

## Prepare data before fitting

Keep participant identifiers, trial labels and provenance in a separate table.
Select features deliberately, enforce consistent units and handle missing
training values outside `PCAPredictor`. Do not insert numeric zeros as a
generic missing-value substitute. Any imputation method must be justified for
the study and fitted using only the training partition.

For motion cycles, choose the cycle boundaries and phase origin, resample
uniformly, and establish how left/right or activity-specific signals are
aligned. The Fourier utility does not perform these steps. Avoid pooling
incompatible coordinate systems or phases simply because their column names
match.

The default variance screen is applied in original units. Inspect retained
features, especially small-amplitude Fourier coefficients and near-zero-mean
signals. Lowering a threshold can retain numerical noise as well as useful
variation.

## Default preprocessing

The default is `sklearn.preprocessing.StandardScaler`. Each selected feature
is centered and scaled independently. This makes the numerical scale of
different units comparable for PCA, but it does not correct data-quality
problems or decide which variables are scientifically meaningful.

Keep this default unless you have a documented reason and validation for
another transform.

## Custom transformer requirements

An optional transformer must provide `fit_transform`, `transform` and
`inverse_transform`. It must preserve shape and column order, be invertible
for the predictions of interest, and operate independently on each feature.
Partial constraints are transformed by filling unconstrained features with
training means. Feature mixing would change a constrained transformed value
according to unspecified inputs.

The code checks methods, shape and per-feature independence using a copied
probe. This finite probe is a compatibility check, not a proof of invertibility
or correctness over the full input domain. Shape-changing transforms, mixed
PCA projections, feature permutations, polynomial expansion and one-hot
encoding are not compatible with the predictor contract.

There is another restriction: prediction and component conversion do not
restore scikit-learn PCA's transformed centering offset. The transformer should
produce centered training features. A generic `MinMaxScaler` may pass the
compatibility probe but produces non-zero feature means, so it is not
generally supported for reliable constrained reconstruction.

## Example: standardized Yeo-Johnson transformation

For suitable complete continuous data, a per-feature Yeo-Johnson transform
with standardization produces centered features:

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer
from sillywalk import PCAPredictor

transformer = make_pipeline(
    PowerTransformer(method="yeo-johnson", standardize=True)
)
model = PCAPredictor(table, transformer=transformer, n_components=0.99)
selected = table.select(model.pca_columns).to_numpy()  # Polars input
transformed_mean = model.transformer.transform(selected).mean(axis=0)
if not np.allclose(transformed_mean, 0, atol=1e-10):
    raise ValueError("Transformer does not center selected features")
```

`table` here must be a prepared Polars dataframe. The transformer is fitted
by the constructor, not separately on the whole study dataset.
Yeo-Johnson can change marginal distributions. It does not guarantee a
Gaussian joint distribution, improve predictive accuracy, separate walking
from running, or resolve an arbitrary nonlinear relationship.

Check constraint residuals, reconstructed values and held-out errors after
choosing a transformer. Box-Cox transforms require strictly positive inputs.
Other transforms can have restricted inverse domains. Fourier coefficients
can be negative, so a positive-only transform may be unsuitable.

## Endpoint correction is a modeling choice

`linear_correction(signal)` adds a ramp so the first and last values equal
their original average. It modifies the entire signal.
`windowed_correction(signal, eps=0.1)` concentrates correction near the ends
and leaves the middle unchanged when a middle interval remains.

Neither function filters noise, resamples time, closes a physical cycle, or
matches endpoint derivatives. Use them only when their change is acceptable
for the study. On an endpoint-exclusive periodic sampling grid, the last
sample need not equal the first. Applying endpoint correction to such an
already-periodic signal can distort it. The synthetic tutorial therefore
does not apply correction.

For measured cycles with a duplicated endpoint, decide and document correction
on that endpoint-inclusive representation, then remove the duplicate before
the FFT. Compare corrected and original signals and quantify the change.
