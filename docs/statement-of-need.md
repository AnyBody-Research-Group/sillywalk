# Statement of need

Researchers preparing musculoskeletal simulations often have a set of motion
profiles and anthropometric measurements, but only some parameters for a new
case. They need a consistent way to represent complete motion cycles, condition
a statistical model on the available measurements, and transfer the resulting
profile into a simulation.

sillywalk connects these steps in Python. Named dataframe columns keep
anthropometric scalars and Fourier coefficients in one feature table.
`PCAPredictor` fits a reduced representation and predicts a complete feature
vector from selected constraints. Fourier utilities produce coefficients using
the convention used by AnyBody's `AnyKinEqFourierDriver`. Jinja templates render
the feature vector as AnyScript.

The intended audience is researchers developing population-based or
parameterized motion workflows and AnyBody users who need scripted parameter
generation. The Python core can also be used for complete numeric tables
without exporting AnyScript.

## Anthropometric profile reconstruction

The authors identify the shared ANSUR adaptation as a way to explore complete
anthropometric profiles conditioned on a subset of measurements. For example,
a researcher with a person's height and selected segment measurements can use
those named values as constraints on a fitted anthropometric model.

Such a profile is a statistical reconstruction, not a newly measured person.
The solver does not constrain results to a convex combination of the training
rows or to observed ranges. An "interpolation" use case therefore still requires
checks for extrapolation, constraint residuals and physical plausibility.
The [dataset guide](external-data.md#shared-ansur-adaptation) records the
author's permission statement and the remaining provenance and unit questions.

## Methodological background

The authors supplied the 2021 Journal of Biomechanics paper
[_Running in circles: Describing running kinematics using Fourier series_](https://doi.org/10.1016/j.jbiomech.2020.110187)
as an earlier application of Fourier representations to running motion.
It provides background for the cyclic-motion representation used here.
This reference is not, by itself, evidence that the current sillywalk package
was used in that study or that its current predictions have been validated.

## Position in the software ecosystem

sillywalk uses NumPy for numerical operations, scikit-learn for PCA and
preprocessing, Narwhals for dataframe access, and Jinja2 for text rendering.
It does not replace these libraries. Its domain-specific contribution is the
combination of constrained reconstruction, motion-feature conventions and
AnyScript output.

Users who need unconstrained dimensionality reduction alone can use
[scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
directly. Users who need missing-training-data estimation, measurement-error
models, probabilistic uncertainty intervals or physical simulation constraints
need methods beyond this implementation. AnyBody performs the musculoskeletal
simulation, not sillywalk.

## What would establish suitability for a study?

A research application should define the source population, units and phase
alignment, train/test split, preprocessing choices, retained component count,
prediction-error metrics and integration checks. For repeated trials, a
participant-level split avoids treating observations from the same person as
independent evidence of generalization.

The included [synthetic tutorial](tutorial.md) makes the software reproducible
without access to a research dataset. It does not establish research adoption or
the accuracy of predicted human motion. Those require measured-data evidence
and author-supplied provenance. The current repository documentation makes no
claim of clinical validation or superiority over alternative approaches.
