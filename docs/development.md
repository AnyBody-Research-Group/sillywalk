# Development and testing

The repository uses Pixi. Run commands from the repository root, with the
manifest and lockfile from the same checkout.

## Run the tests

```console
pixi run --locked test
```

For a specific module:

```console
pixi run test tests/test_mlpca.py
```

On Windows the same selection can be written as `tests\test_mlpca.py`.
The test task invokes pytest. Appended arguments are forwarded to pytest.

The `default` environment uses Python 3.14. Separate environments are available
for Python 3.11, 3.12, 3.13 and 3.14:

```console
pixi run --locked -e py311 test
pixi run --locked -e py313 test
pixi run --locked test-coverage
```

The repository CI matrix covers Python 3.11, 3.12 and 3.13 on Linux,
Windows and macOS runners. Environment availability alone is not evidence of
successful execution on every target. Inspect the
[CI runs](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/ci.yml)
for the actual result.

The ordinary suite covers selected PCA fitting/prediction behavior, component
conversion, trusted persistence, custom transformer validation and
Fourier/text-rendering behavior. It excludes the `anybody` marker, so the
cross-platform Python tests do not require a proprietary console or license.
The notebook's standalone export test uses `write_anyscript` and pytest's
temporary directory. Its historical `Fourier.csv` fixture uses AMMR 3 paths:
the test converts period `T` to frequency and makes driver references
body-model-relative. That test checks rendering, not AMMR 4 compatibility.

## Optional real AnyBody load test

On Windows, the separate `anybody` environment includes Python 3.13,
AnyBody Console and AnyPyTools. The committed lockfile records Console
8.2.1 and AnyPyTools 1.20.6. Use that lockfile for local reproduction:

```console
pixi install --locked -e anybody
```

You also need an appropriately licensed AMMR 4 checkout. The reference beta
revision is `89c8730d7b7df7f74b4eca5f6c5cd0f5e72f7ded` from
[AMMR4-Beta](https://github.com/AnyBody/AMMR4-Beta), which identifies itself as
`4.0.1-beta`. Read and accept the vendor's terms yourself and confirm that
your AnyBody license permits use of this beta. Those requirements are separate
from sillywalk's MIT license.

After doing so, a shallow pinned test dependency can be obtained with:

```console
git init AMMR4-Beta
git -C AMMR4-Beta fetch --depth=1 https://github.com/AnyBody/AMMR4-Beta.git 89c8730d7b7df7f74b4eca5f6c5cd0f5e72f7ded
git -c core.longpaths=true -C AMMR4-Beta checkout --detach FETCH_HEAD
```

Configure a valid AnyBody license outside the repository using the console's
normal licensing settings. The test inherits the current process environment
and lets the console resolve its license. It does not retrieve credentials.
Do not put license passwords in repository files, command-line arguments or
issue reports. A network license also requires access to its license server.

Set the AMMR directory and an optional persistent log directory in the current
PowerShell process, then select the real test:

```powershell
$env:AMMR_PATH = "C:\work\AMMR4-Beta"
$env:SILLYWALK_ANYBODY_LOG_DIR = "C:\work\sillywalk-anybody-load-logs"
pixi run --locked -e anybody test-anybody
```

`test-anybody` runs pytest directly. Missing dependencies, AMMR configuration
or a usable license fail the selected run rather than skip it.
An alternative console executable can be selected with `ANYBODYCON_PATH`;
otherwise the test uses the environment's actual `AnyBodyCon.exe`, not a
shell wrapper. The dedicated command disables AnyPyTools' separate pytest
collection plugin to avoid an extra console process for version discovery.
It still uses the real `AnyPyProcess` API for loading and querying the model.

To launch the console interactively with the same environment:

```console
pixi run -e anybody anybodycon
```

A startup banner or a missing-registry-key warning alone does not establish
whether a license was acquired or a model loaded. Inspect the actual load result.

The test predicts a deterministic synthetic profile, calls the public
`write_anyscript(..., create_human_model=True, ammr_version=(4, 0, 0))`, and
uses real AnyPyTools `Load` and `Dump` commands. It checks the loaded study's
`nStep`, height, mass, period and knee-driver coefficient arrays. A local
`libdef.any` shim points to `AMMR_PATH` without changing AMMR or the console's
installation.

The output directory contains `predicted.main.any`, the shim,
`anybody-load.log`, and `loader-result.json`. Console output is quiet.
The retained log replaces license values present in the environment with
`[redacted]`. The JSON contains error/warning counts rather than raw
diagnostics, version identifiers and finite numeric `Dump` values. Its
`dump_validation_succeeded` field becomes `true` only after all six expected
values pass their comparisons. Treat logs
as local diagnostic material and inspect them before sharing: exact-value
redaction is not a guarantee that all sensitive information is removed.
The log directory's named files are overwritten on a repeated run. If it is
not set, pytest's temporary directory is used.

This test exercises model loading only. It does not run kinematics or inverse
dynamics and does not validate measured-data prediction or physical plausibility.

## Trusted current-version AnyBody CI

The separate
[AnyBody compatibility workflow](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/anybody.yml)
selects the current published Console and AnyPyTools releases on a Windows
runner. It runs on pushes to `main`, on Mondays at 03:29 UTC, or through
manual dispatch. Its job runs only in this repository on the default branch.
Pull requests, fork repositories and manual dispatches from other branches
cannot run the secret-bearing job.

CI starts with the committed integration environment, then runs:

```console
pixi update --environment anybody anybodycon anypytools --no-config --tls-root-certs system
```

This updates only the selected packages and necessary integration dependencies
in the runner's working copy. It does not commit a new lockfile or update the
ordinary test environments. CI checks the resolved versions against the
public conda registries' latest releases and fails if the environment cannot
resolve those releases. It then fetches the current `ammr4-beta` branch of
[AMMR4-Beta](https://github.com/AnyBody/AMMR4-Beta) at the exact SHA resolved
at the start of the run. This means the newest available pair, not stable
AMMR 3 or every historical combination.

The repository owner must maintain qualifying vendor licensing and two
repository Actions secrets: `RLM_LICENSE` for the license server endpoint
and `RLM_LICENSE_PASSWORD` for its password. Only the load-test step receives
those values. Missing secrets fail that step, and unreachable servers,
unavailable licenses or model-loading failures fail the selected test.
The beta's license and retained `LICENSE` file remain separate from
sillywalk's MIT license. Do not run this workflow with an untrusted code
revision or add its licensing settings to ordinary pull-request CI.

The workflow artifact contains only our generated `predicted.main.any`,
`resolved-versions.json` and, when the console produced version evidence,
`loader-result.json`. It records the sillywalk commit, Python and package
versions, AMMR SHA and the runtime console version/build. The runtime
build comes from the console banner, not the conda package build string.
On failure the numeric report remains unsuccessful, or is absent if setup
failed before it could be written. No artifact uploads console logs, the
AMMR checkout, the path-bearing `libdef.any` shim or credentials.

A passing run establishes only the generated synthetic model's `Load` and
six numerical `Dump` checks for that resolved pair. It does not establish
all-version compatibility, kinematics, dynamics or biomechanical validity.
The workflow has been authored but not dispatched during preparation.
Earlier local attempts with Console 8.2.1.14738/build 20634.23340,
AnyPyTools 1.20.6 and AMMR 4.0.1-beta failed during licensing before
loading. See the [integration guide](anybody-integration.md) for that boundary
and the unresolved proprietary access question for reviewers.

## Run the offline example

```console
pixi run --locked -e docs docs-example
```

The example checks its analytic Fourier coefficients, signal reconstruction,
height constraint residual and trusted persistence round trip. It writes a
versioned numerical report without downloading data. See the
[tutorial](tutorial.md) for its output contract.

## Build documentation

```console
pixi run --locked -e docs docs-build
```

This runs `zensical build --strict` using `zensical.toml` and writes `site`.
Open `site\index.html` on Windows, or `site/index.html` on Linux/macOS.
The site uses file-style links so it can be read without a web server. No
hosted documentation URL or deployment is assumed.
System fonts avoid remote font downloads. Browser restrictions can prevent
interactive search from working under `file://`. The pages and navigation
remain readable, or use the optional local preview for search.

The
[Documentation workflow](https://github.com/AnyBody-Research-Group/sillywalk/actions/workflows/docs.yml)
runs the synthetic example and builds the site. The built site is uploaded as a
workflow artifact. This workflow does not publish to GitHub Pages.

If you want an interactive local preview, run this yourself:

```console
pixi run -e docs docs-serve
```

Stop it when finished. A preview server is not needed for the offline tutorial
or static build.

## Formatting and linting

```console
pixi run -e lint lint
```

The task runs all configured pre-commit hooks, including Ruff, formatting,
type checking and text checks. `pre-commit-run` is an alias:

```console
pixi run -e lint pre-commit-run
```

Hooks can modify files. Review the diff after running them.
Do not assume a formatted result is a successful scientific validation.

## Environment maintenance

Intentional dependency changes should update both `pixi.toml` and `pixi.lock`.
Do not install packages manually inside `.pixi` environments and then rely
on that unrecorded state for reproducibility.

If a local Pixi mirror configuration points to an unavailable network,
diagnose it before changing the project. Current Pixi supports a per-command
`--no-config` option to ignore inherited user/system configuration. Do not
disable TLS verification or replace the repository lockfile merely to
hide an installation problem.

For contribution, support and issue-reporting procedures see
[CONTRIBUTING.md](https://github.com/AnyBody-Research-Group/sillywalk/blob/main/CONTRIBUTING.md).
Report the package versions, smallest reproducible input, expected behavior
and actual error without sharing restricted participant data.
