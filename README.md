# sillywalk

[![CI](https://img.shields.io/github/actions/workflow/status/AnyBody-Research-Group/sillywalk/ci.yml?style=flat-square&branch=main)](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)
[![pypi-version](https://img.shields.io/pypi/v/sillywalk.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/sillywalk)
[![python-version](https://img.shields.io/pypi/pyversions/sillywalk?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/sillywalk)

`sillywalk` is a Python library for Maximum Likelihood Principal Component Analysis (ML-PCA). It allows you to build statistical models from data and predict missing values based on observed values. While it can be used with any numerical dataset, it includes special utilities for working with data from the [AnyBody Modeling System™](https://www.anybodytech.com/).

## Installation

You can install `sillywalk` from PyPI,

```bash
pip install sillywalk
```

or as a conda package:

```bash
pixi install sillywalk
```

### For developers

This project is managed by [pixi](https://pixi.sh). To set up a development environment:

```bash
git clone https://github.com/AnyBody-Research-Group/sillywalk
cd sillywalk

pixi install
pixi run test
```

## Quick Start

Here's a quick example of how to use `sillywalk` to predict missing data.

### 1. Create a statistical model

First, you need a dataset to build the model. `sillywalk` works with both Pandas and Polars DataFrames.

```python
import pandas as pd
import sillywalk

# Sample data of student measurements
data = {
    "Sex": [1, 1, 2, 2, 1, 2],
    "Age": [25, 30, 28, 22, 35, 29],
    "Stature": [175, 180, 165, 160, 185, 170],
    "Bodyweight": [70, 80, 60, 55, 85, 65],
    "Shoesize": [42, 44, 39, 38, 45, 40],
}
df = pd.DataFrame(data)

# Create a PCAPredictor model from the data
model = sillywalk.PCAPredictor(df)
```

### 2. Predict missing values

Once the model is created, you can use it to predict missing values based on some known values (constraints).

```python
# Define the known values (constraints)
constraints = {
    "Stature": 180,
    "Bodyweight": 65,
}

# Predict the missing values
result = model.predict(constraints)

# The result is a dictionary containing the original constraints
# and predicted values for the other variables.
print(result)
```

### 3. Save and load the model

You can save your trained model to a file and load it later to make new predictions without having to re-train it.

```python
# Save the model to a file
model.save_npz("student_model.npz")

# Load the model from the file
loaded_model = sillywalk.PCAPredictor.from_npz("student_model.npz")

# Use the loaded model to make predictions
new_constraints = {"Age": 24, "Shoesize": 43}
prediction = loaded_model.predict(new_constraints)

print(prediction)
```

## Working with output from AnyBody Models

`sillywalk` can work with time series data from AnyBody and convert it to fourier coefficients
that are compatible with AnyBody's `AnyKinEqFourierDriver` class:

```python
import polars as pl
import numpy as np
import sillywalk

# Example time series data (e.g., joint angles over a gait cycle)
time_points = np.linspace(0, 1, 101)  # 101 time points
hip_flexion = 30 * np.sin(2 * np.pi * time_points) + 10
knee_flexion = 60 * np.sin(2 * np.pi * time_points + np.pi/4)

# Create DataFrame with time series
time_series_df = pl.DataFrame({
    'Main.HumanModel.BodyModel.Interface.Trunk.PelvisThoraxExtension': hip_flexion,
    'Main.HumanModel.BodyModel.Interface.Right.KneeFlexion': knee_flexion,
})

# Convert to fourier coefficients (default n_modes=6)
fourier_df = sillywalk.anybody.compute_fourier_coefficients(time_series_df, n_modes=6)

┌────────────┬────────────┬────────────┬───────────┬───┬───────────┬───────────┬───────────┬───────────┐
│ ...tension ┆ ...tension ┆ ...tension ┆ ...tensio ┆ … ┆ ...Flexio ┆ ...Flexio ┆ ...Flexio ┆ ...Flexio │
│ _a0        ┆ _a1        ┆ _a2        ┆ n_a3      ┆   ┆ n_b2      ┆ n_b3      ┆ n_b4      ┆ n_b5      │
│ ---        ┆ ---        ┆ ---        ┆ ---       ┆   ┆ ---       ┆ ---       ┆ ---       ┆ ---       │
│ f64        ┆ f64        ┆ f64        ┆ f64       ┆   ┆ f64       ┆ f64       ┆ f64       ┆ f64       │
╞════════════╪════════════╪════════════╪═══════════╪═══╪═══════════╪═══════════╪═══════════╪═══════════╡
│ 10.0       ┆ 0.928198   ┆ -0.025044  ┆ -0.021042 ┆ … ┆ -0.550711 ┆ -0.307976 ┆ -0.218252 ┆ -0.169925 │
└────────────┴────────────┴────────────┴───────────┴───┴───────────┴───────────┴───────────┴───────────┘
```

The two time series columns will be decomposed into each (`n_modes*2-1`) fourier coefficients.
Each coefficient will have the suffix `_a0` to `a_5` and `_b1` to `_b5`. The value of `_b0` is not
stored as this is always 0.

### Generating AnyBody models from data

Sillywalk can create anyscript include-files, from a dictionary (dataframe with one row) with fourier coefficients and/or other anthropometric data.

Fourier drivers are create if the keys/columns follow a specific format: `DOF:<measure>[<idx>]_<a/b><mode>`. For example:

> `DOF:Main.HumanModel.BodyModel.Main.HumanModel.BodyModel.Interface.Trunk.PelvisPosX[0]_a4`

The dataset also needs to have a single value `T`, which is period of the fourier driver (until the motion repeats again).

The include file is generated by calling `sillywalk.anybody.create_model_file()`:

```python

# Generate an AnyBody include file from the predicted data
sillywalk.anybody.create_model_file(
    predicted_data,
    targetfile="predicted_motion.any"
)
```

The generated file will contain `AnyKinEqFourierDriver` entries that can be included in your AnyBody model:

```anyscript
AnyFolder PCA_drivers = {
  AnyVar Freq =  DesignVar(5.6674/(2*pi));

  AnyFolder JointsAndDrivers = {
    AnyKinEqFourierDriver Trunk_PelvisPosX_Pos_0 = {
      Type = CosSin;
      Freq = ..Freq;
      AnyKinMeasure &m = Main.HumanModel.BodyModel.Main.HumanModel.BodyModel.Interface.Trunk.PelvisPosX;
      AnyVar a0_offset ??= DesignVar(0.0);
      A = {{  0 + a0_offset, 0.000740, -0.00194, -1.8560e-05, 1.7458e-05, -4.8094e-05,  }};
      B = {{ 0, 0.00068, 0.00044, -0.000153, -0.000205, -1.02480e-07,  }};
    };
    ...
```

### Creating Complete Human Models

You can also generate a complete stand-alone AnyBody model:

```python
# Generate a complete model file with human model template
sillywalk.anybody.create_model_file(
    predicted_data,
    targetfile="complete_human_model.any",
    create_human_model=True
)
```
