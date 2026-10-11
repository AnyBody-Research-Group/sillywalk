# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Added a draft JOSS paper, documentation, an offline synthetic workflow with reproducibility tests, contributor guidance, citation metadata, and issue templates.
- Corrected Fourier coefficient output construction to return one row while preserving the pandas or Polars backend, including Polars 2.0.
- Reject missing PCA eigenvectors or eigenvalues when loading model data, and isolate notebook persistence tests in temporary directories.
- Replace the notebook's obsolete model-export call with a collected `write_anyscript` regression using temporary output. Add an opt-in Windows AnyBody Console load test; real loading requires a usable license and access to the configured license server.
- Add trusted default-branch Windows CI selecting the latest available Console and AnyPyTools with the current AMMR 4 beta revision. Keep the committed local environment for reproduction and retain only generated-model, numeric and version evidence as artifacts. Workflow execution and successful licensed loading remain pending.
- Add the confirmed software author's ORCID and document the linked Zenodo release archive without assigning an anticipated DOI.

## [1.3.0] - 2026-05-03

- **Added:** `PCAPredictor` now supports advanced custom preprocessing transformers (e.g., `MinMaxScaler`, `PowerTransformer`, or a `Pipeline`) via the `transformer` argument, enabling tailored normalization before ML-PCA while preserving compatibility with partial-constraint prediction when using element-wise compatible transforms.

## [1.2.0] - 2026-04-27

- **Fix:** `mlpca` — correct principal-component orientation and derive retained component count properly, so projections and reconstructions work after truncation.
- **Fix:** `mlpca` — detect infeasible collinear constraints and raise a clear `ValueError` instead of silently dropping conflicting constraints.
- **Fix:** `mlpca` — make `pca_explained_variance_ratio` reflect truncation (sums < 1 when components are dropped) and persist/load it in `.npz` exports.
- **Fix:** `mlpca` — floor tiny/zero PCA eigenvalues to avoid inf/NaN in the KKT solve, warn when clamping occurs, and rename local variable to `n_components` for clarity.
- **Tests:** Added regression tests covering zero eigenvalues, truncation round-trips, and infeasible-constraint detection.

## [1.1.1] - 2025-12-28

- Updated the pelvis offset code in the template used for creating AnyBody models.

## [1.1] - 2025-12-25

### Changed

- Use `freq` column in data (instead of the period `T`) when constructing the drivers. `freq` has a direct correlation with speed which is often used as constraint in the walking/running models. This is a breaking change, since AnyBody data needs to contain a `freq` column instead of the `T` column.

- Update include logic for `libdef.any` in the templated for generated AnyBody models.

## [1.0.2] - 2025-12-20

### Fixed

- Fix calculation of relative ratios when calculating which columns to exclude from PCA.

## [1.0.1] - 2025-12-11

### Added

- LICENSE file is now included in the source distribution

## [1.0.0] - 2025-12-11

### Added

- Initial release
