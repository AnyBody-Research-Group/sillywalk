# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
