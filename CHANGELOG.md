# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
