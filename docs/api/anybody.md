# Fourier and AnyScript reference

```python
from sillywalk import anybody
```

These functions operate on arrays, dataframes and local files. None invokes the
AnyBody Modeling System.

## `linear_correction`

```python
anybody.linear_correction(y)
```

Takes a one-dimensional iterable or NumPy signal. Returns a float NumPy array
with a linear ramp added so both endpoints equal the original endpoint average.
For fewer than two elements it returns a copy. All samples may change.

```python
corrected = anybody.linear_correction([0.0, 1.0, 2.0, 3.0])
```

It does not match derivatives or validate physical periodicity.

## `windowed_correction`

```python
anybody.windowed_correction(y, eps=0.1)
```

Returns a float NumPy array with endpoint correction tapered near the first
and last `eps` fraction of samples. The middle is unchanged when a middle
interval remains. Short arrays return a copy. Nonfinite or nonpositive `eps`
falls back to linear correction, values above `0.5` are capped, and a window
covering the whole array also falls back to linear correction.

```python
corrected = anybody.windowed_correction(signal, eps=0.1)
```

Both correction utilities expect one-dimensional signals. They are not a
replacement for selecting and phase-aligning a cycle.

## `compute_fourier_coefficients`

```python
anybody.compute_fourier_coefficients(df_native, n_modes=6)
```

Takes a dataframe whose columns are finite numeric uniformly sampled periodic
signals. Every supplied column is processed, so remove time and metadata
columns first. Returns a one-row dataframe using the input's native backend.

For each original name `signal`, it emits:

```text
signal_a0, signal_a1, ..., signal_a[n_modes-1]
signal_b1, ..., signal_b[n_modes-1]
```

`n_modes` must be an integer at least 1. Values below 1 raise `ValueError`.
If fewer real-FFT bins exist than requested, only available modes are emitted.
The coefficient units are the original signal units. The constant `a0` is the
signal mean and sine coefficients use the positive-sine reconstruction
convention.

```python
import numpy as np
import polars as pl

phase = np.arange(128) / 128
signals = pl.DataFrame(
    {"DOF:Interface.Right.KneeFlexion.Pos[0]": 0.4 + 0.2 * np.sin(2 * np.pi * phase)}
)
coefficients = anybody.compute_fourier_coefficients(signals, n_modes=3)
```

No frequency scalar is inferred. Add `freq=1/period_seconds` when preparing
export data. See [Data conventions](../data-conventions.md#periodic-signal-tables)
for endpoint and Nyquist cautions.

## `write_anyscript`

```python
anybody.write_anyscript(
    data,
    targetfile="trialdata.any",
    template_file=None,
    prepfunc=...,  # Built-in grouping function when omitted.
    create_human_model=False,
    ammr_version=None,
)
```

| Argument             | Meaning                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `data`               | Flat dictionary of string keys to numeric values. A dataframe is not accepted directly.                                    |
| `targetfile`         | String/`Path` output filename. Parent directories are created. Existing file is overwritten. `None` returns rendered text. |
| `template_file`      | Optional local UTF-8 Jinja template filename. Default is the packaged `model.any.jinja`.                                   |
| `prepfunc`           | Optional callable that converts the flat mapping to a template context, replacing the built-in grouping step.              |
| `create_human_model` | Wrap the include in a minimal human-model scaffold when `True`. Default `False`.                                           |
| `ammr_version`       | Optional `(major, minor, patch)` tuple passed to the template. Omission uses a limited naming heuristic.                   |

Returns `None` when writing a file or a string when `targetfile=None`.
Convert a one-row Polars coefficient dataframe to a dictionary before export:

```python
data = coefficients.row(0, named=True)
data["freq"] = 1.2
text = anybody.write_anyscript(data, targetfile=None)
```

The default template renders driver groups only with the exact `DOF:` prefix.
Use measure paths relative to `Main.HumanModel.BodyModel`, and include an
explicit `.Pos[index]` suffix. Sparse coefficient indices are zero-filled.
Unrecognized scalar names are not rendered automatically.

The template context has `scalar_data` and `fourier_data`. Each Fourier group
contains `prefix`, `measure`, `index`, `a` and `b`.
`create_human_model` and `ammr_version` are added by the writer.

The function does not validate finite numbers, positive frequency, AnyBody
syntax or physical completeness. Missing template variables may render as
empty text. Use trusted dictionaries/templates and inspect the result.
See [AnyBody integration](../anybody-integration.md) for version selection,
driver exclusions and the external execution checklist.

## Public scope

`linear_correction`, `windowed_correction`,
`compute_fourier_coefficients` and `write_anyscript` are the documented user
utilities. Underscore-prefixed helpers are implementation details.
Older examples calling `create_model_file` or passing `template=` do not
match the current public API.
