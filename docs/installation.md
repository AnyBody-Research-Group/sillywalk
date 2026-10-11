# Installation

## Requirements

The package requires Python 3.11 or newer. Its runtime dependencies are NumPy,
scikit-learn, Narwhals and Jinja2. Dataframes are supplied by the caller:
install pandas or Polars for the examples below. A bare NumPy matrix is not a
supported `PCAPredictor` input. Wrap it in a named dataframe first.

The source workspace locks environments for Windows x86-64, Linux x86-64 and
macOS Apple Silicon. These lockfile targets are not a promise that every Python
version and platform combination has been tested. See
[Development and testing](development.md) for the CI matrix.

You do not need AnyBody to fit, predict, calculate Fourier coefficients or
render files. AnyBody and AMMR are required only to load and execute AnyScript.
They are not installed by this package.

## Recommended: run from the source checkout with Pixi

Install [Pixi](https://pixi.sh/latest/installation/) and Git, then:

```console
git clone https://github.com/AnyBody-Research-Group/sillywalk.git
cd sillywalk
pixi run --locked -e docs docs-example
```

The `docs` environment contains Python 3.13, the local sillywalk package, Polars
and Zensical. Pixi creates the environment on first use using `pixi.lock`.
Initial setup needs network access to download dependencies. Once installed,
the synthetic example itself has no network requests.

Pixi must satisfy `requires-pixi` in `pixi.toml` (currently at least 0.76.0).
The workspace uses the `pixi-build` preview feature to build the local package
and a version 7 lockfile.
Use a current Pixi release if an older release cannot read the manifest or
lockfile. No system Python is needed for this workflow.

All repository Python commands should run through Pixi:

```console
pixi run python
pixi run -e docs python -m examples.synthetic_workflow --output-dir example-output
```

## Install the published package

For use outside this repository, create a Python environment and install
sillywalk plus a dataframe backend:

```console
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```sh
. .venv/bin/activate
```

Then install:

```console
python -m pip install sillywalk polars
```

Use `pandas` instead of `polars` if that is your dataframe backend. The published
package version can differ from the checkout version. Record the exact package
versions used for research. The complete example is a repository module,
not part of the installed `sillywalk` library. A package-only installation
cannot run `examples.synthetic_workflow`. Clone the repository and run from its
root to use that example.

In an existing Pixi project, published conda packages can be added with:

```console
pixi add sillywalk polars
```

Or create a conda environment:

```console
conda create -n sillywalk -c conda-forge python=3.13 sillywalk polars
conda activate sillywalk
```

Registry availability and versions are independent of the source lockfile.
If you need the code documented here, use the checkout workflow above.

## Check the installation

From this repository:

```console
pixi run -e docs python -c "import sillywalk; print(sillywalk.__version__)"
pixi run -e docs docs-example
```

A successful example prints the location of `report.json` and states that
AnyBody was not invoked. See the [tutorial](tutorial.md#outputs) for the files
you should find.

## Troubleshooting

| Symptom                               | Check                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `pixi` is not recognized              | Install Pixi and open a new terminal so its executable is on `PATH`.                                                      |
| `No module named polars` or `pandas`  | Use `-e docs` for the example, or install the chosen backend in your own project.                                         |
| `No module named examples`            | Run from the cloned repository root, not a directory containing only the installed package.                               |
| Locked environment resolution fails   | Check your Pixi version, platform, network access and whether `pixi.toml` and `pixi.lock` belong to the same checkout.    |
| PCA rejects NaN or infinity           | Prepare a complete finite numeric matrix. Partial prediction inputs are not missing training values.                      |
| An exported `.any` file does not load | Python rendering is not AnyBody validation. Follow the [integration checklist](anybody-integration.md#verify-in-anybody). |
