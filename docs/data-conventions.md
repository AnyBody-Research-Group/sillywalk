# Data conventions

## Predictor tables

`PCAPredictor` takes an eager, named dataframe, such as a pandas or Polars
dataframe. Each row is one observation and each column is one feature. A row
might represent a trial, a participant summary or a simulated case. The
software does not infer which of these it is.

For time-series workflows, convert each complete cycle to one row of Fourier
coefficients, then join the corresponding scalars. Do not pass a long table of
timestamps as if each timestamp were an independent participant.

| Input                            | Convention                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Column names                     | Exact, case-sensitive strings. Prediction constraints must name selected PCA columns.                        |
| Numeric training values          | Complete, finite values. No automatic imputation of selected features.                                       |
| Units                            | Consistent across training, constraints and outputs. The predictor does not convert units.                   |
| Identifiers and categories       | Keep metadata separately. Numeric category codes are treated as continuous quantities.                       |
| Constant or low-variance columns | Excluded according to screening thresholds and returned at their training means. They cannot be constrained. |
| Nonnumeric columns               | Replaced internally by NaN for numeric reductions and returned as NaN, not their original text.              |
| Prediction input dataframe       | Exactly one row intended. The current converter uses the first row, rather than providing batch prediction.  |

A NumPy matrix alone has no column names and is not a supported constructor
input. Wrap it in a dataframe:

```python
import numpy as np
import polars as pl
from sillywalk import PCAPredictor

values = np.array([[1.65, 61.0], [1.72, 70.0], [1.80, 79.0]])
table = pl.DataFrame(values, schema=["Height", "Weight"], orient="row")
model = PCAPredictor(table, n_components=1)
prediction = model.predict({"Height": 1.75})
```

Do not interpret encoded `Sex`, activity labels or subject numbers as supported
categorical modeling. A continuous PCA prediction of such a column can be
fractional. A variable called `Stature` is not automatically equivalent to
`Height`, and centimetres are not converted to metres.

For the [shared ANSUR adaptation](external-data.md#shared-ansur-adaptation),
select documented continuous measurements explicitly. Exclude identifiers,
revision metadata, acquisition fields and encoded categories even when some
are stored as numbers. A model-conditioned anthropometric profile does not
guarantee interpolation within observed ranges or membership of the source
population.

## Periodic signal tables

`compute_fourier_coefficients` takes one uniformly sampled cycle per column,
with rows in phase order. Every column supplied is transformed. Exclude
timestamps, phase columns and metadata before calling it.

The function does not use a time coordinate, resample, filter or align cycles.
If samples cover one period `T`, use the phase grid `j/N` for
`j=0,...,N-1`. The corresponding time grid is `T*j/N`. Omit a duplicated final
endpoint. Frequency is a separate scalar, `freq=1/T`, measured in cycles per
second (Hz), not angular frequency in radians per second.

The retained representation is:

```text
x(phase) = a0 + sum_j [aj*cos(2*pi*j*phase) + bj*sin(2*pi*j*phase)]
```

Here `j` starts at 1. `a0` is the mean, not twice the mean. The sine sign follows
the positive-sine convention in this equation. There is no emitted `b0`.
`n_modes=6` means the constant and harmonics 1 through 5, for 11 columns per
input signal when enough FFT bins are available.

Coefficients have the same units as the original signal. Angular AnyBody
measures normally use radians and translations metres. Check the actual target
measure instead of assuming a unit from its name.

For even sample counts, the current FFT scaling also doubles the Nyquist
coefficient. Avoid retaining that highest bin (keep `n_modes <= N/2`) unless
you have explicitly checked reconstruction. The tutorial uses only low
harmonics, well below Nyquist.

## Names for the built-in AnyScript template

The coefficient calculator appends `_a0`, `_a1`, ... and `_b1`, ... to the
original column name. It does not add a `DOF:` prefix or rewrite AnyBody paths.
The exporter groups these suffixes, and its built-in template renders drivers
only for groups with the exact prefix `DOF:`.

Use a path relative to `Main.HumanModel.BodyModel`, with an explicit
`.Pos[index]` suffix to select a component:

```text
DOF:Interface.Right.KneeFlexion.Pos[0]_a0
DOF:Interface.Right.KneeFlexion.Pos[0]_a1
DOF:Interface.Right.KneeFlexion.Pos[0]_b1
```

The template renders the reference
`Main.HumanModel.BodyModel.Interface.Right.KneeFlexion` and
`MeasureOrganizer = {0}`. Providing the full
`Main.HumanModel.BodyModel...` path currently duplicates that prefix.

These names illustrate the export contract, not a guarantee that the measure
exists in every AMMR release. Verify target paths in your own model.

| Scalar key recognized by the default template         | Meaning                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| `Height`                                              | `Main.HumanModel.Anthropometrics.BodyHeight`, conventionally m           |
| `Weight`                                              | `Main.HumanModel.Anthropometrics.BodyMass`, conventionally kg            |
| `freq`                                                | Positive cycle frequency in Hz, used to render `Period=1/freq`           |
| `Main.HumanModel.Anthropometrics.SegmentDimensions.*` | Named segment dimensions, conventionally m                               |
| `Main.ModelSetup.TrialSpecificData.*`                 | Named trial-specific values, with units defined by the receiving model   |
| `Main.ModelSetup.SubjectSpecificData.*`               | Named subject-specific values, with units defined by the receiving model |

Other scalar keys are available to a custom template but are not automatically
rendered by the default template. For example, `omega`, `Bodyweight` and
`Stature` are not aliases for `freq`, `Weight` and `Height`.
