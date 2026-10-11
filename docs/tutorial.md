# Offline synthetic workflow

This tutorial uses generated signals and generated anthropometry. It needs no
remote dataset, participant data, network service or AnyBody license after the
Python environment is installed.

The source is
[examples/synthetic_workflow.py](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/examples/synthetic_workflow.py).
It is importable without running the workflow or writing files.

## Run it

From the repository root:

```console
pixi run --locked -e docs docs-example
```

To choose an output directory:

```console
pixi run --locked -e docs python -m examples.synthetic_workflow --output-dir example-output
```

Or invoke it from Python in the source checkout:

```python
from pathlib import Path
from examples.synthetic_workflow import run

paths = run(Path("example-output"))
print(paths["report"])
```

The output directory is created if necessary. The example overwrites its named
output files on repeated runs. Use a fresh directory if you want to keep an
earlier run.

A successful run exits with status 0 and prints these two lines (the path
separator follows your operating system):

```text
Synthetic workflow complete. Report: example-output\report.json
AnyScript rendered only; AnyBody was not invoked.
```

If a numerical check fails, the command raises an error and exits with a
nonzero status. Output files alone are not proof that all checks passed.

## What the example does

### 1. Generate complete training observations

A NumPy random generator with seed `2026` creates 48 pairs of latent values.
Each pair produces a synthetic height, body mass, cycle frequency and knee
angle signal. Every signal has 128 uniformly spaced phase samples:

```python
phase = np.arange(128) / 128
```

The interval is `[0, 1)`. The endpoint of the period is deliberately omitted
because it would duplicate the first periodic sample.

Each signal is a sum of a constant, two cosine harmonics and two sine
harmonics. Its coefficients depend on the latent values. This is a
constructed two-dimensional relationship, not an empirical relationship
between stature and human movement.

### 2. Compute Fourier features

```python
from sillywalk import anybody

coefficients = anybody.compute_fourier_coefficients(
    signal_dataframe, n_modes=3
)
```

`n_modes=3` emits `a0`, `a1`, `a2`, `b1`, `b2` for each signal. The example
uses the name `DOF:Interface.Right.KneeFlexion.Pos[0]`. This includes the
prefix needed by the built-in exporter and a path relative to
`Main.HumanModel.BodyModel`. See [Data conventions](data-conventions.md).

The training table contains one row per generated observation, not one row
per time sample. It has eight columns: `Height`, `Weight`, `freq` and five
Fourier coefficients. Height is in metres, mass in kilograms, frequency in
Hz and angular signals in radians.

### 3. Fit and predict from one partial constraint

```python
from sillywalk import PCAPredictor

model = PCAPredictor(training, n_components=2)
prediction = model.predict({"Height": 1.76})
```

Two principal components retain the constructed latent structure. Prediction
returns every training column, including the Fourier coefficients and frequency
needed for file export. It is a model-conditioned synthetic profile, not an
"average person" or a population estimate.

For this feasible example, the height residual must be at most `1e-9` m.
Do not assume every set of constraints in a truncated model is feasible.
Check residuals against your own requirements.

### 4. Save and reload a trusted model

```python
model.export_pca_data("example-output/synthetic_model.npz")
loaded = PCAPredictor.from_pca_data("example-output/synthetic_model.npz")
reloaded_prediction = loaded.predict({"Height": 1.76})
```

The example reloads only the file it has just written. All predicted values
must agree within absolute tolerance `1e-12`.

Do not load untrusted `.npz` model files. The archive contains a pickled fitted
transformer and the loader deserializes it. A model file is executable,
trusted input, not a safe data exchange format.

### 5. Render an AnyScript include

```python
anybody.write_anyscript(
    prediction, targetfile="example-output/predicted_motion.any"
)
```

The example checks that a driver and the intended measure reference are
present. It does not load the file in AnyBody. A single knee driver does not
constrain a complete human model. The output must be integrated into a
compatible model with its remaining drivers and setup.

## Outputs

| File                       | Contents                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------- |
| `training_data.csv`        | 48 synthetic observations with named scalar and Fourier features                          |
| `signals.csv`              | Phase and the first synthetic knee signal, in radians                                     |
| `fourier_coefficients.csv` | One row of coefficients for that first signal                                             |
| `synthetic_model.npz`      | Fitted model and serialized transformer, for trusted local reuse                          |
| `prediction.json`          | Predicted scalar and Fourier values for `Height=1.76` m                                   |
| `predicted_motion.any`     | Rendered include containing anthropometry and one Fourier driver                          |
| `report.json`              | Seed, dimensions, units, software versions, numerical residuals and verification boundary |

The first signal's coefficients must match their analytic values within
`1e-12`. Reconstructing that signal from the retained harmonics must also agree
within `1e-12`. These checks establish Fourier behavior for this deliberately
band-limited signal only. They do not measure held-out prediction performance.

Numerical residuals and PCA signs may differ across numerical-library builds.
Compare shapes, tolerances and reconstructed outputs rather than requiring
byte-identical archives.

The report lists Python-package versions only. It contains no executed AnyBody
or AMMR version evidence. If you integrate the output in AnyBody, preserve a
separate execution record with the receiving-model revision and results.

## Adapt this to measured data

Keep subject identifiers and provenance outside the predictor's numeric matrix.
Align cycle phase, establish units, and validate preprocessing on the training
partition only. Split repeated observations by participant when assessing
generalization to new people. Select the retained components using an explicit
validation plan.

Before exporting measured predictions, check all constraint residuals, physical
plausibility, positive frequency and the target AnyBody measure paths. See
[Methods and limitations](methods.md), [Preprocessing](preprocessing.md) and
[AnyBody integration](anybody-integration.md).
