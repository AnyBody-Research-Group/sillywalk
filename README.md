# sillywalk

[![CI](https://img.shields.io/github/actions/workflow/status/AnyBody-Research-Group/sillywalk/ci.yml?style=flat-square&branch=main)](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)

<!-- [![conda-forge](https://img.shields.io/conda/vn/conda-forge/sillywalk?logoColor=white&logo=conda-forge&style=flat-square)](https://prefix.dev/channels/conda-forge/packages/sillywalk)
[![pypi-version](https://img.shields.io/pypi/v/sillywalk.svg?logo=pypi&logoColor=white&style=flat-square)](https://pypi.org/project/sillywalk)
[![python-version](https://img.shields.io/pypi/pyversions/sillywalk?logoColor=white&logo=python&style=flat-square)](https://pypi.org/project/sillywalk) -->

Library for maximum likelihood principal-component-analysis for AnyBody models

The project can also be used on ml-pca on any type of data, but does have some extra
utility functions which makes it easier to work with data from AnyBody models.

## Installation

This project is managed by [pixi](https://pixi.sh).
You can install the package in development mode using:

### From source

```bash
git clone https://github.com/AnyBody-Research-Group/sillywalk
cd sillywalk

pixi run pre-commit-install
pixi run test
```

#### With pixi

```bash
pixi init my-new-project
cd my-new-project
pixi add pixi add https://prefix.dev/anybody-beta::sillywalk
```
