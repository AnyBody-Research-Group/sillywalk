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
