# sillywalk

[![CI](https://img.shields.io/github/actions/workflow/status/AnyBody-Research-Group/sillywalk/ci.yml?style=flat-square&branch=main)](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)
[![pypi-version](https://img.shields.io/pypi/v/sillywalk.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/sillywalk)
[![python-version](https://img.shields.io/pypi/pyversions/sillywalk?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/sillywalk)

sillywalk is a Python library for PCA-based prediction of motion and
anthropometric parameters from partial inputs, with Fourier utilities and
AnyScript export for the AnyBody Modeling System.

Researchers can represent each motion cycle as a row of Fourier coefficients,
combine it with named anthropometric measurements, fit a reduced statistical
model and reconstruct a complete profile from selected constraints. The library
uses NumPy, scikit-learn, Narwhals and Jinja2 rather than replacing their
numerical or dataframe functionality.

The Python workflow does not require AnyBody. Loading or running generated
AnyScript requires a separately installed AnyBody Modeling System and compatible
AMMR. Rendered files are not automatically validated musculoskeletal models.
The optional [AnyBody integration checks](docs/anybody-integration.md) target
selected console and AMMR versions. They do not establish compatibility with
every release or validate kinematics and dynamics.

## Start with the offline example

Install [Pixi](https://pixi.sh/latest/installation/), then:

```console
git clone https://github.com/AnyBody-Research-Group/sillywalk.git
cd sillywalk
pixi run --locked -e docs docs-example
```

This creates 48 seeded synthetic observations, fits a model, predicts a profile
from a height constraint, checks model persistence and Fourier reconstruction,
and writes an AnyScript include to `example-output`.

The example has no dataset downloads or AnyBody execution. Initial environment
installation needs network access. Generated data illustrates software behavior,
not measured human motion or evidence of predictive accuracy.
See the [offline tutorial](docs/tutorial.md) for inputs, output files and units.

## Install the library

Python 3.11 or newer is required. In your own Python environment:

```console
python -m pip install sillywalk polars
```

Use pandas instead of Polars if preferred. In an existing Pixi project,
`pixi add sillywalk polars` installs the published conda package.
See [installation instructions](docs/installation.md) for source environments,
conda, platform targets and troubleshooting.

## Minimal prediction

```python
import polars as pl
from sillywalk import PCAPredictor

# Constructed values: height in metres, mass in kilograms.
table = pl.DataFrame({
    "Height": [1.60, 1.68, 1.76, 1.84],
    "Weight": [60.0, 68.0, 76.0, 84.0],
})
model = PCAPredictor(table, n_components=1)
prediction = model.predict({"Height": 1.72})

model.export_pca_data("synthetic_model.npz")
loaded = PCAPredictor.from_pca_data("synthetic_model.npz")  # Trusted local file only.
```

Selected training features must be complete and finite. Missing inputs are
supported as partial prediction constraints, not as missing training values.
The implementation fits scikit-learn PCA, not an incomplete-data
maximum-likelihood PCA estimator. Check constraint residuals and physical
plausibility before using a prediction.

Model archives contain a pickled fitted transformer. Loading an untrusted
`.npz` model can execute code. Use trusted artifacts only.

## Documentation

| Topic                                             | Guide                                                                           |
| ------------------------------------------------- | ------------------------------------------------------------------------------- |
| Research scope and intended use                   | [Statement of need](docs/statement-of-need.md)                                  |
| Reproducible end-to-end workflow                  | [Offline tutorial](docs/tutorial.md)                                            |
| Names, dataframe shapes, units and cycle sampling | [Data conventions](docs/data-conventions.md)                                    |
| PCA, constraints and verification limits          | [Methods and limitations](docs/methods.md)                                      |
| Custom transformers and endpoint correction       | [Preprocessing](docs/preprocessing.md)                                          |
| Fourier drivers and AnyBody requirements          | [AnyBody integration](docs/anybody-integration.md)                              |
| Public API                                        | [PCAPredictor](docs/api/predictor.md), [AnyBody utilities](docs/api/anybody.md) |
| Dataset provenance and reuse boundaries           | [External datasets](docs/external-data.md)                                      |
| Tests, linting and documentation build            | [Development](docs/development.md)                                              |

Build the navigable documentation site locally:

```console
pixi run --locked -e docs docs-build
```

Open `site\index.html` on Windows, or `site/index.html` on Linux/macOS.
The documentation CI builds an artifact without publishing a website.

## Development, support and citation

```console
pixi run --locked test
pixi run -e lint lint
```

Report problems through the
[issue tracker](https://github.com/AnyBody-Research-Group/sillywalk/issues).
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution and support procedures,
[CITATION.cff](CITATION.cff) for citation metadata, and [LICENSE](LICENSE) for
the MIT license. Dataset and proprietary software licenses are separate.

The [JOSS preparation folder](joss-prep/README.md) records submission-readiness
work and questions that require author answers.

For methodological background, the authors supplied the 2021 paper
[_Running in circles: Describing running kinematics using Fourier series_](https://doi.org/10.1016/j.jbiomech.2020.110187).
The [dataset guide](docs/external-data.md) records their free-use statement for
shared data, including the ANSUR adaptation, while retaining the remaining
source, unit and reuse-term questions. Neither that statement nor the method
paper is evidence of measured-data predictive accuracy or independent package
adoption.
