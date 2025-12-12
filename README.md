# sillywalk

[![CI](https://img.shields.io/github/actions/workflow/status/AnyBody-Research-Group/sillywalk/ci.yml?style=flat-square&branch=main)](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)
[![pypi-version](https://img.shields.io/pypi/v/sillywalk.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/sillywalk)
[![python-version](https://img.shields.io/pypi/pyversions/sillywalk?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/sillywalk)

sillywalk is a Python library for statistical modeling of human motion and anthropometric data with the AnyBody Modeling System. It implements Maximum Likelihood Principal Component Analysis (ML‑PCA) to learn compact, low‑dimensional models from datasets, predict missing or individualized signals from partial inputs, and export those predictions as AnyScript include files that plug directly into AnyBody models.

Key features

- AnyBody I/O and preprocessing: Post‑process AnyBody time series and convert them to Fourier coefficients compatible with `AnyKinEqFourierDriver`.
- ML‑PCA modeling and prediction: Fit ML‑PCA models from tabular data, handle missing values naturally, and predict new samples from partial constraints; save/load models to and from `.npz`.
- AnyBody model generation: Generate templated AnyScript include files (e.g., drivers and optional human model blocks) from predicted Fourier coefficients and anthropometry.
- Friendly data interfaces: Works with pandas or polars DataFrames and NumPy arrays; installable via PyPI or pixi for reproducible workflows.

See Quick Start below for a minimal end‑to‑end example.

## Installation

With [pixi](https://pixi.sh):

```bash
pixi add sillywalk
```

or from PyPI:

```bash
pip install sillywalk
```

or with conda:

```bash
conda create -n sillywalk -c conda-forge sillywalk
conda activate sillywalk
```

### Developer Setup

This project uses `pixi` for dependency management and development tools.

```bash
git clone https://github.com/AnyBody-Research-Group/sillywalk
cd sillywalk
pixi install
pixi run test
```

See [pixi documentation](https://pixi.sh/latest/) for more info.

---

## Quick Start

### 1. Build a Model

```python
import pandas as pd
import sillywalk

data = {
    "Sex": [1, 1, 2, 2, 1, 2],
    "Age": [25, 30, 28, 22, 35, 29],
    "Stature": [175, 180, 165, 160, 185, 170],
    "Bodyweight": [70, 80, 60, 55, 85, 65],
    "Shoesize": [42, 44, 39, 38, 45, 40],
}
df = pd.DataFrame(data)
model = sillywalk.PCAPredictor(df)
```

### 2. Predict Missing Values

```python
constraints = {"Stature": 180, "Bodyweight": 65}
result = model.predict(constraints)
```

### 3. Save and Load Models

```python
model.export_pca_data("student_model.npz")
loaded = sillywalk.PCAPredictor.from_pca_data("student_model.npz")
prediction = loaded.predict({"Age": 24, "Shoesize": 43})
```

---

## AnyBody Model Utilities

`sillywalk` can convert time series data to Fourier coefficients compatible with AnyBody's `AnyKinEqFourierDriver`:

```python
import polars as pl
import numpy as np
import sillywalk

time = np.linspace(0, 1, 101)
hip = 30 * np.sin(2 * np.pi * time) + 10
knee = 60 * np.sin(2 * np.pi * time + np.pi/4)

df = pl.DataFrame({
    'Main.HumanModel.BodyModel.Interface.Trunk.PelvisThoraxExtension': hip,
    'Main.HumanModel.BodyModel.Interface.Right.KneeFlexion': knee,
})

fourier_df = sillywalk.anybody.compute_fourier_coefficients(df, n_modes=6)
print(fourier_df)
```

Each time series column is decomposed into Fourier coefficients (`_a0` to `_a5`, `_b1` to `_b5`).

```
┌────────────┬────────────┬───────────┬───┬───────────┬───────────┬───────────┐
│ ...tension ┆ ...tension ┆ ...tensio ┆ … ┆ ...Flexio ┆ ...Flexio ┆ ...Flexio │
│ _a0        ┆ _a1        ┆ n_a2      ┆   ┆ n_b3      ┆ n_b4      ┆ n_b5      │
│ ---        ┆ ---        ┆ ---       ┆   ┆ ---       ┆ ---       ┆ ---       │
│ f64        ┆ f64        ┆ f64       ┆   ┆ f64       ┆ f64       ┆ f64       │
╞════════════╪════════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╡
│ 10.0       ┆ 0.928198   ┆ -0.021042 ┆ … ┆ -0.550711 ┆ -0.218252 ┆ -0.169925 │
└────────────┴────────────┴───────────┴───┴───────────┴───────────┴───────────┘
```

### Generate AnyBody Include Files

You can generate AnyScript include files from a dictionary or DataFrame with Fourier coefficients and anthropometric data:

```python
sillywalk.anybody.write_anyscript(
    predicted_data,
    targetfile="predicted_motion.any"
)
```

This creates `AnyKinEqFourierDriver` entries for use in AnyBody models.

#### Example: Complete Human Model

```python
sillywalk.anybody.write_anyscript(
    predicted_data,
    targetfile="complete_human_model.any",
    create_human_model=True
)
```

---

## PCAPredictor

PCAPredictor selects numeric columns with sufficient variance and fits a PCA model. It can:

- Predict all columns from partial constraints on PCA columns using a KKT least‑squares system.
- Convert between primal parameters and principal components.
- Persist models to `.npz` files.

Notes

- Constraints on columns excluded from PCA are not allowed and raise ValueError.
- If no constraints are provided, `predict` returns the column means.
- If no columns pass variance screening, the model has zero components and `predict` returns means.

Example

```python
import pandas as pd
from sillywalk import PCAPredictor

df = pd.DataFrame({
    "a": [1, 2, 3, 4],
    "b": [2, 2.5, 3, 3.5],
    "c": [10, 10, 10, 10],  # excluded (zero variance)
})
model = PCAPredictor(df)
pred = model.predict({"a": 3.2})
pcs = model.parameters_to_components({k: pred[k] for k in model.pca_columns})
back = model.components_to_parameters(pcs)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
