# Contributing to sillywalk

Contributions can improve the Python API, numerical tests, AnyScript templates,
documentation, or examples. You do not need access to the proprietary AnyBody
Modeling System to work on or test the Python library.

## Get help or report a problem

Use the [public issue tracker](https://github.com/AnyBody-Research-Group/sillywalk/issues)
for bug reports, usage questions, and proposed improvements. Search existing
issues first. A useful report includes the sillywalk version, Python version,
operating system, a small reproducible example, the expected result, and the
actual result or full traceback.

For a numerical problem, include the input shape, column names and units,
constraints, preprocessing choices, and tolerances. Replace participant data
with synthetic inputs where possible. Do not post personal information,
credentials, restricted datasets, or proprietary model files. If an example
needs AnyBody, also identify the AnyBody and AnyBody Managed Model Repository
(AMMR) versions and the relevant model configuration.

The issue tracker is also the place to ask whether a proposed change fits the
project. Support is provided through public discussion as contributor
availability permits. There is no guaranteed response time or support
agreement.

## Set up a development checkout

Install [pixi](https://pixi.sh/latest/installation/) and Git, then run:

```sh
git clone https://github.com/AnyBody-Research-Group/sillywalk.git
cd sillywalk
pixi install --locked
pixi run test
```

Run commands from the repository root. `pixi.lock` records the development
dependencies for Windows, Linux, and macOS on Apple silicon. The first install
requires network access to obtain packages. Python and the development tools
run inside pixi environments, so a separate system Python is not required.

The default environment includes the package and test dependencies. To run a
focused check or test another supported Python version:

```sh
pixi run test tests/test_mlpca.py -q
pixi run -e py311 test
```

CI runs the test suite with Python 3.11, 3.12, and 3.13 on Windows, Linux, and
macOS. See the [CI workflow](.github/workflows/ci.yml) for the current matrix.
The separate [AnyBody workflow](.github/workflows/anybody.yml) runs only
trusted default-branch code with repository licensing secrets. It checks
one current Console/AMMR 4 beta pair, not all versions or complete
kinematics. Fork pull requests do not receive those secrets. See
[development guidance](docs/development.md) for local reproduction,
vendor requirements and the limited numerical evidence.

Install the lint environment and run the repository's pre-commit hooks:

```sh
pixi install --locked -e lint
pixi run lint
```

The hooks check formatting, imports, types, spelling, and file hygiene. Some
hooks modify files. Inspect the resulting diff and rerun the command before
submitting a change. To run individual checks without rewriting files:

```sh
pixi run -e lint ruff check src tests examples
pixi run -e lint typecheck-python
```

## Propose a change

Fork the repository and make changes on a branch in your fork. Keep a pull
request focused on one problem. Include the motivation, relevant issue links,
an explanation of changed behavior, and the checks you ran. For an API or
numerical change, describe compatibility implications and add a regression
test. Update the affected documentation and examples.

Prefer existing helpers and patterns over new dependencies. Discuss substantial
API changes, algorithm changes, or new dependencies in an issue before
implementing them. Pull requests and issue discussions document the reasons
for design decisions. Repository maintainers decide whether to merge a
change. Opening a pull request does not guarantee inclusion or a release
date.

Use constructive, respectful language in issues and reviews. Critique the
code or method rather than the person. This guidance does not appoint an
enforcement team or establish a separate Code of Conduct policy.

Contributions are made under the project's [MIT license](LICENSE). Only
submit code, text, data, and figures that you have permission to distribute,
and identify any third-party material and its license.

## Make scientific examples reproducible

The [synthetic workflow](examples/synthetic_workflow.py) provides an offline
Python example once dependencies are installed:

```sh
pixi run python examples/synthetic_workflow.py --output-dir example-output
pixi run test tests/test_synthetic_workflow.py -q
```

It generates synthetic inputs, fits a PCA predictor, saves and reloads the
model, computes Fourier coefficients, and writes an AnyScript include file.
The output report records the workflow's numerical checks. This verifies
Python-side calculations and generated text, not execution of AnyBody or
validation against measured human motion.

For a reproducible analysis, record the software version or commit, pixi
environment, random seed, input provenance and license, units, preprocessing,
retained PCA components, constraints, and numerical tolerances. Keep inputs
and generated outputs separate. Tests should write to pytest's temporary
directories rather than overwrite repository files.

Use deterministic synthetic fixtures for routine tests. Check numerical
properties such as constrained values, saved-model predictions, Fourier
amplitudes, and the required driver structure. Do not replace these checks
with a test that only confirms a script exits successfully. Dataset downloads
and proprietary AnyBody runs should remain explicit, optional workflows,
with their extra requirements documented.

Only load `.npz` model files from trusted sources. The saved transformer is
deserialized with Python pickle, which can execute code when a model is
loaded. Reproduce examples by generating their model locally.

## Cite the software

Use [CITATION.cff](CITATION.cff) for the software citation and state the exact
version or commit used in your methods. GitHub's "Cite this repository"
feature can export a citation from that file. Until a paper has been
published, cite the software rather than treating the draft manuscript as a
published article.

For a publication archive, the release owner should:

1. Select the reviewed software version and verify that `pyproject.toml`,
   `pixi.toml`, `CITATION.cff`, and the release notes agree on that version.
2. Complete the relevant tests and documentation checks for the chosen
   commit, then create a tagged release through the project's release
   process.
3. The repository owner has linked GitHub releases to Zenodo. Check that
   the tagged release produced a completed Zenodo deposit, confirm its
   metadata and deposited files, and obtain the version-specific software
   DOI. If deposit fails, resolve it before citing the archive.
4. Add only the confirmed archive identifier and release date to
   `CITATION.cff`. For reproducibility, distinguish a DOI for one release
   from a concept DOI that refers to the software across releases.

For JOSS, coordinate the final release and archive with the editor's
instructions. Preparing these files does not create a release, register a
DOI, or submit a paper. If a paper is accepted and published, add its
confirmed citation and DOI to `CITATION.cff` without discarding the software
version and archive information.
