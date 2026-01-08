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
import polars as pl
import sillywalk

data = {
    "Sex": [1, 1, 2, 2, 1, 2],
    "Age": [25, 30, 28, 22, 35, 29],
    "Stature": [175, 180, 165, 160, 185, 170],
    "Bodyweight": [70, 80, 60, 55, 85, 65],
    "Shoesize": [42, 44, 39, 38, 45, 40],
}
df = pl.DataFrame(data)
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

You can generate AnyScript include files from a dictionary or DataFrame with Fourier coefficients and/or anthropometric data:

Let us try to generate a model from an anthropometric dataset. First we will download the data

```python
>>> df =  pl.read_parquet("https://anybodydatasets.blob.core.windows.net/sillywalk/kso-running/kso-running-fourier-2025-12-28-0.parquet")
>>> df
shape: (114, 1_317)
┌──────────┬────────────┬────────────┬───────────┬───┬───────────┬───────────┬───────────┬───────────┐
│ freq     ┆ Main.Human ┆ Main.Human ┆ Main.Huma ┆ … ┆ CenterOfM ┆ CenterOfM ┆ CenterOfM ┆ CenterOfM │
│ ---      ┆ Model.Anth ┆ Model.Anth ┆ nModel.An ┆   ┆ ass.PosZ_ ┆ ass.PosZ_ ┆ ass.PosZ_ ┆ ass.PosZ_ │
│ f64      ┆ ropometric ┆ ropometric ┆ thropomet ┆   ┆ b2        ┆ b3        ┆ b4        ┆ b5        │
│          ┆ …          ┆ …          ┆ ric…      ┆   ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
│          ┆ ---        ┆ ---        ┆ ---       ┆   ┆ f64       ┆ f64       ┆ f64       ┆ f64       │
│          ┆ f64        ┆ f64        ┆ f64       ┆   ┆           ┆           ┆           ┆           │
╞══════════╪════════════╪════════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪═══════════╡
│ 1.415094 ┆ 0.164147   ┆ 0.107101   ┆ 0.119941  ┆ … ┆ 0.000389  ┆ -0.000774 ┆ 0.00013   ┆ 0.000322  │
│ 1.395349 ┆ 0.164147   ┆ 0.107101   ┆ 0.119941  ┆ … ┆ 0.000571  ┆ -0.000821 ┆ 0.000059  ┆ 0.000282  │
│ 1.382488 ┆ 0.164147   ┆ 0.107101   ┆ 0.119941  ┆ … ┆ -0.000097 ┆ -0.000827 ┆ 0.000112  ┆ 0.000301  │
│ 1.395349 ┆ 0.164147   ┆ 0.107101   ┆ 0.119941  ┆ … ┆ 0.000522  ┆ -0.000882 ┆ 0.000209  ┆ 0.000253  │
│ 1.369863 ┆ 0.164147   ┆ 0.107101   ┆ 0.119941  ┆ … ┆ 0.000482  ┆ -0.000949 ┆ 0.000141  ┆ 0.000215  │
│ …        ┆ …          ┆ …          ┆ …         ┆ … ┆ …         ┆ …         ┆ …         ┆ …         │
│ 1.321586 ┆ 0.1782     ┆ 0.116442   ┆ 0.13021   ┆ … ┆ 0.000114  ┆ -0.000281 ┆ -0.000132 ┆ 0.000007  │
│ 1.327434 ┆ 0.1782     ┆ 0.116442   ┆ 0.13021   ┆ … ┆ 0.00038   ┆ -0.000289 ┆ -0.000112 ┆ -0.000051 │
│ 1.382488 ┆ 0.16105    ┆ 0.107724   ┆ 0.117678  ┆ … ┆ -0.000125 ┆ -0.00011  ┆ 0.000099  ┆ -0.000029 │
│ 1.428571 ┆ 0.16105    ┆ 0.107724   ┆ 0.117678  ┆ … ┆ -0.000062 ┆ -0.000198 ┆ -0.000006 ┆ -0.00008  │
│ 1.485149 ┆ 0.16105    ┆ 0.107724   ┆ 0.117678  ┆ … ┆ 0.000787  ┆ -0.000113 ┆ 0.000108  ┆ 0.000021  │
└──────────┴────────────┴────────────┴───────────┴───┴───────────┴───────────┴───────────┴───────────┘
```

This dataset contains data from a 114 running subjects, with their 'AnyBody' anthropometic dimensions and fourier coefficients to recreated their running patterns.

Sillywalk can create PCA model from this data set, and we can get a prediction of average person with a height of 1.8m.

```python
>>> model = sillywalk.PCAPredictor(df)
>>> predicted_data = model.predict({"Height": 1.8})
>>> pl.DataFrame(predicted_data)
shape: (1, 1_317)
┌──────────┬────────────┬────────────┬───────────┬───┬───────────┬───────────┬───────────┬───────────┐
│ freq     ┆ Main.Human ┆ Main.Human ┆ Main.Huma ┆ … ┆ CenterOfM ┆ CenterOfM ┆ CenterOfM ┆ CenterOfM │
│ ---      ┆ Model.Anth ┆ Model.Anth ┆ nModel.An ┆   ┆ ass.PosZ_ ┆ ass.PosZ_ ┆ ass.PosZ_ ┆ ass.PosZ_ │
│ f64      ┆ ropometric ┆ ropometric ┆ thropomet ┆   ┆ b2        ┆ b3        ┆ b4        ┆ b5        │
│          ┆ …          ┆ …          ┆ ric…      ┆   ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
│          ┆ ---        ┆ ---        ┆ ---       ┆   ┆ f64       ┆ f64       ┆ f64       ┆ f64       │
│          ┆ f64        ┆ f64        ┆ f64       ┆   ┆           ┆           ┆           ┆           │
╞══════════╪════════════╪════════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪═══════════╡
│ 1.395979 ┆ 0.170115   ┆ 0.112083   ┆ 0.124302  ┆ … ┆ -0.000044 ┆ -0.000206 ┆ -7.6567e- ┆ 0.000057  │
│          ┆            ┆            ┆           ┆   ┆           ┆           ┆ 8         ┆           │
└──────────┴────────────┴────────────┴───────────┴───┴───────────┴───────────┴───────────┴───────────┘
```

Sillywalk can then create an AnyBody model for us using this data:

```
>>> sillywalk.anybody.write_anyscript(
...     predicted_data,
...     targetfile="predicted_motion.any"
... )
```

This creates `AnyKinEqFourierDriver` entries for use in AnyBody models.

```anyscript

Main.HumanModel.Anthropometrics =
{
  BodyMass = 75.61897598069909;
  BodyHeight = 1.8;
};

Main.HumanModel.Anthropometrics.SegmentDimensions =
{
  PelvisWidth = DesignVar(0.17011475699044465);
  PelvisHeight = DesignVar(0.11208281142857143);
  PelvisDepth = DesignVar(0.12430194406161754);

...


AnyFolder PCA_drivers = {
  AnyVar Period =  DesignVar(1/1.3959786246912738);

  AnyFolder JointsAndDrivers = {
    AnyKinEqFourierDriver Trunk_PelvisPosX_Pos_0 = {
      Type = CosSin;
      Freq = 1/..Period;
      CType = {Hard};
      Reaction.Type = {Off};
      MeasureOrganizer = {0};
      AnyKinMeasure &m = Main.HumanModel.BodyModel.Main.HumanModel.BodyModel.Interface.Trunk.PelvisPosX;
      AnyVar a0_offset ??= DesignVar(0.0);
      A = {{  8.623540080091557e-10 + a0_offset, 0.00017990165903964024, 0.011945580192769307, -0.00013312081080007037, 0.00041700269239085025, -7.518299551487054e-05,  }};
      B = {{ 0, 0.0004029987590162868, -0.010429170745175874, 0.00024439360060977196, 0.0003970888221702516, 5.258816436360581e-05,  }};
    };
    ...

```

#### Example: Complete Human Model

Sillywalk will by default generate 'anyscript' files with antrhopometics and drivers which can be included in other models. But it is also possible to create a complete standalone model.

```python
sillywalk.anybody.write_anyscript(
    predicted_data,
    targetfile="complete_human_model.any",
    create_human_model=True
)
```

or using a jinja template for complete control:

```python
sillywalk.anybody.write_anyscript(
    predicted_data,
    targetfile="complete_human_model.any",
    template="MyModel.any.jinja",
    create_human_model=True
)
```

See the [template sillywalk](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/src/sillywalk/templates/model.any.jinja) uses as example.

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
import polars as pl
from sillywalk import PCAPredictor

df = pl.DataFrame({
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
