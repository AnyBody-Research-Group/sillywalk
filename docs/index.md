# sillywalk

sillywalk is a Python library for PCA-based prediction of motion and
anthropometric parameters from partial inputs. It converts periodic signals to
Fourier coefficients and renders predicted parameters as AnyScript include
files for the AnyBody Modeling System.

The Python workflow runs without AnyBody. Loading or running the generated
AnyScript requires a separately installed AnyBody Modeling System and a
compatible AnyBody Managed Model Repository (AMMR).

## Start with the offline example

[Install the source environment](installation.md), then run this command from
the repository root:

```console
pixi run -e docs docs-example
```

This creates a seeded synthetic dataset, fits a model, predicts from a height
constraint, checks a trusted save/load round trip, and writes Fourier
coefficients and `predicted_motion.any`. No remote data or proprietary software
is used. Follow the [tutorial](tutorial.md) to inspect the inputs and outputs.

The example demonstrates software behavior. It is not measured human data,
evidence of predictive accuracy, or a validated musculoskeletal model.

## Find the right guide

| Goal                                                 | Read                                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------------------ |
| Understand the research problem                      | [Statement of need](statement-of-need.md)                                |
| Prepare data with consistent names, units and phases | [Data conventions](data-conventions.md)                                  |
| Understand fitting and constrained prediction        | [Methods and limitations](methods.md)                                    |
| Choose preprocessing safely                          | [Preprocessing and transformers](preprocessing.md)                       |
| Render files for an existing AnyBody model           | [AnyBody integration](anybody-integration.md)                            |
| Use the core public API                              | [PCAPredictor](api/predictor.md) and [AnyBody utilities](api/anybody.md) |
| Run checks or contribute                             | [Development and testing](development.md)                                |

## Scope and evidence

The estimator fits scikit-learn PCA on a complete numeric training matrix.
Partial constraints are supported at prediction time. It does not implement
missing-data maximum-likelihood fitting, despite the historical ML-PCA
terminology in the package description.

The project does not establish clinical applicability, biomechanical validity,
or performance superiority over other methods. The authors have stated that
their shared data, including an ANSUR adaptation, is free to use. The
[dataset guide](external-data.md) separates that permission statement from
the source, unit and formal reuse terms still needing documentation.
The supplied [running Fourier study](statement-of-need.md#methodological-background)
is methodological background, not independent evidence of package adoption.
See the [limitations](methods.md#limitations) before using predictions in a
research analysis.

## Support and citation

Report reproducible problems in the
[issue tracker](https://github.com/AnyBody-Research-Group/sillywalk/issues).
For contribution procedures, citation metadata and the MIT license, see
[CONTRIBUTING.md](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/CONTRIBUTING.md),
[CITATION.cff](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/CITATION.cff)
and [LICENSE](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/LICENSE).
Documentation sources are included in the repository and can be built locally.
