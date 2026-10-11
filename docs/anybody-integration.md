# AnyBody integration

## What is implemented

The library's Python utilities compute coefficients and render AnyScript.
They do not launch AnyBody, validate an AMMR path, solve kinematics or execute
inverse dynamics. You can run all Python onboarding without proprietary
software. A separate, opt-in integration test launches the real console
through AnyPyTools. See [the development guide](development.md#optional-real-anybody-load-test)
for its Windows environment, licensing prerequisites and AMMR setup.

To load or run the output, you need a separately licensed
[AnyBody Modeling System](https://www.anybodytech.com/software/) installation
and a compatible [AMMR](https://anyscript.org/ammr/).
Exact verified versions, supported anatomical measure sets and executed
reference models still need author confirmation.

The loading check attempted on 2026-10-11 used AnyBody Console
`8.2.1.14738` (runtime build `20634.23340`), AnyPyTools `1.20.6` and AMMR
`4.0.1-beta`, pinned to commit
`89c8730d7b7df7f74b4eca5f6c5cd0f5e72f7ded`.
The license settings were supplied to the test process, but the configured
license endpoint was unreachable in a bounded TCP check and the console
reported "No license available". It did not reach model loading or produce
the requested `Dump` values. These versions describe the attempted check,
not a verified compatibility combination.

## Automated compatibility checks

The authors requested CI checks against the newest available AnyBody console
and AMMR. This is a maintenance policy, not evidence that every released
AnyBody/AMMR combination works. The configured
[AnyBody compatibility workflow](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/.github/workflows/anybody.yml)
selects the newest console package and AnyPyTools release available to its
environment, together with the current `ammr4-beta` branch revision of
AMMR4-Beta. It records the resolved versions and pins the resolved AMMR commit
for that run. This AMMR 4 beta branch is the selected target, not a search of
every AMMR release. The locked local environment provides a separate
reproducible reference combination.

The workflow runs on trusted default-branch updates, a weekly schedule or a
default-branch manual invocation, not on pull requests with license secrets.
It preserves the generated model and selected numeric/version evidence.
AMMR source, the local AMMR path shim and raw console logs are not uploaded
as workflow artifacts.

The real integration test loads one generated synthetic model and queries its
loaded study settings, anthropometry, period and knee-driver coefficient
arrays using `Dump`. A passing run demonstrates those assertions for that
generated fixture and selected environment only. It does not check all
templates, anatomical measures or release combinations, run kinematics or
inverse dynamics, or establish predictive or physical validity.

Workflow configuration and test collection are not successful execution
evidence. This workflow has not been executed as part of this preparation,
and its configured licensing remains unverified. The license-blocked attempt
above remains the observed local result.
See [Development and testing](development.md#optional-real-anybody-load-test)
for the selected test, configuration and CI evidence policy. The selected
integration test fails when its required environment or usable license is
missing rather than reporting a skipped test as compatibility.

## Reviewer access

Reviewers can reproduce the fitting, prediction, persistence, Fourier and
text-rendering workflows without proprietary software. Verifying the real
AnyBody loading claims requires a usable AnyBody license and compatible AMMR.
The authors have left reviewer access arrangements unresolved. Do not assume
reviewers have that access or treat the Python tests as a substitute. The
authors need to agree the evaluation scope and access arrangements with the
JOSS editor.

## Render an include for an existing model

```python
from pathlib import Path
from sillywalk import anybody

data = {
    "Height": 1.76,
    "Weight": 75.0,
    "freq": 1.2,
    "DOF:Interface.Right.KneeFlexion.Pos[0]_a0": 0.4,
    "DOF:Interface.Right.KneeFlexion.Pos[0]_a1": 0.08,
    "DOF:Interface.Right.KneeFlexion.Pos[0]_b1": 0.2,
}
anybody.write_anyscript(data, targetfile=Path("predicted_motion.any"))
```

This produces anthropometric assignments and a `PCA_drivers` folder with a
`CosSin` Fourier driver. The reference uses a path relative to
`Main.HumanModel.BodyModel`, and `freq` is in Hz.

In your model's appropriate scope, include the generated file:

```anyscript
#include "predicted_motion.any"
```

Where the include belongs, which existing drivers it replaces, and which study
uses `PCA_drivers` are model-specific decisions. The tutorial's single knee
driver is not a complete set of human-body constraints.

The default template sets hard kinematic constraints and turns driver reactions
off. It also emits an `AnyMechObjectExcluder` for selected rhythm drivers, using
AMMR-dependent paths. Review these choices rather than appending generated
drivers blindly to an existing model.

## AMMR version selection

`ammr_version=(major, minor, patch)` is passed to the template. If omitted,
the implementation guesses only from two specific, fully qualified lumbar
coefficient keys:

```text
AMMR 4 marker:
DOF:Main.HumanModel.BodyModel.Trunk.Joints.Lumbar.SacrumPelvis.Pos[0]_a0

AMMR 3 marker:
DOF:Main.HumanModel.BodyModel.Trunk.JointsLumbar.SacrumPelvisJnt.Pos[0]_a0
```

If neither is present, no version is inferred and the template's older-version
branch is used. This is a naming heuristic, not a compatibility test.
These full-path marker names also conflict with the body-model-relative paths
expected by the emitted driver references. Prefer an explicit `ammr_version`
and verified relative paths instead of relying on the guess.

The offline example leaves the version unspecified to demonstrate current file
rendering. It does not establish support for the resulting default rhythm paths.

## Optional standalone wrapper

```python
anybody.write_anyscript(
    data,
    targetfile=Path("synthetic.main.any"),
    create_human_model=True,
    ammr_version=(4, 0, 0),  # Select only for your verified target installation.
)
```

This wraps the output in a `Main` model that includes an AMMR human model and a
kinematic study. It looks for `libdef.any` locally, then in the installation's
AMMR directory. The built-in wrapper disables several muscle groups and adds
camera/video setup.

"Standalone wrapper" means a generated text scaffold, not a fully constrained,
validated model or an inverse-dynamics workflow. A complete compatible
coefficient set and model setup are still needed. Do not infer executed
compatibility with AMMR 4.0.0 from the illustrative tuple above.

## Custom templates

Use the actual keyword `template_file`, not the obsolete `template` keyword:

```python
text = anybody.write_anyscript(
    data,
    targetfile=None,
    template_file="my_model.any.jinja",
    ammr_version=(4, 0, 0),
)
```

A template receives `scalar_data`, `fourier_data`, `create_human_model` and
`ammr_version`. Each Fourier group contains `prefix`, `measure`, `index`,
`a` and `b`. A custom `prepfunc` can replace the grouping step. Inspect the
[built-in template](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/src/sillywalk/templates/model.any.jinja)
when creating a model-specific exporter.

Only use trusted templates and trusted keys. Jinja rendering is not a sandbox
or an AnyScript syntax validator. The built-in Jinja environment can render
missing values as empty text. A missing `freq` can therefore yield an invalid
period expression without a Python error. Validate finite coefficients, positive
frequency and intended output explicitly before handing the file to AnyBody.

## Verify in AnyBody

1. Record the AnyBody Modeling System version, AMMR version and receiving-model
   revision. Resolve all measure paths in that model.
2. Check units, cycle frequency, phase origin, component indices and coefficient
   completeness. Inspect the generated `A` and `B` arrays.
3. Check the include scope, anthropometric assignments, driver exclusions and
   conflicts with existing drivers. Supply all remaining required constraints.
4. Load the model. Resolve warnings and errors rather than treating text export
   as a completed integration.
5. Execute the intended kinematic operation. Examine initialization, closure,
   joint trajectories and physical plausibility.
6. If the study uses dynamics, contact or muscle recruitment, validate those
   separately in the complete receiving model. Preserve the model files and
   executed results alongside the research record.

The repository's Python tests and synthetic example cannot replace these
checks. No AnyBody load or execution is claimed by the offline workflow.

Keep external execution evidence in the study's reproducibility record, with
the receiving-model source/revision, generated include, AnyBody/AMMR versions,
operation logs and inspected results. The synthetic `report.json` records
Python versions and numerical checks only, not this external evidence.
