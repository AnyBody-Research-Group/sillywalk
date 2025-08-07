# sillywalk

[![CI](https://img.shields.io/github/actions/workflow/status/AnyBody-Research-Group/sillywalk/ci.yml?style=flat-square&branch=main)](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)
[![pypi-version](https://img.shields.io/pypi/v/sillywalk.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/sillywalk)
[![python-version](https://img.shields.io/pypi/pyversions/sillywalk?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/sillywalk)

**sillywalk** is a Python library for Maximum Likelihood Principal Component Analysis (ML-PCA). It enables you to build statistical models from data and predict missing values based on observed values. While it is general-purpose, it includes special utilities for working with data from the [AnyBody Modeling System™](https://www.anybodytech.com/), 

## Installation

Install from PyPI:

```bash
pip install sillywalk
```

Or with [pixi](https://pixi.sh):

```bash
pixi install sillywalk
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
model.save_npz("student_model.npz")
loaded = sillywalk.PCAPredictor.from_npz("student_model.npz")
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
sillywalk.anybody.create_model_file(
    predicted_data,
    targetfile="predicted_motion.any"
)
```

This creates `AnyKinEqFourierDriver` entries for use in AnyBody models.

#### Example: Complete Human Model

```python
sillywalk.anybody.create_model_file(
    predicted_data,
    targetfile="complete_human_model.any",
    create_human_model=True
)
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

