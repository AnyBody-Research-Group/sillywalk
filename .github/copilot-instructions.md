This repository uses the `pixi` package manager to manage dependencies and development tools.
Please refer to the [`pixi` documentation](https://pixi.readthedocs.io/en/latest/) for more information on how to use it.
Please also refer to the [`pixi` source code](https://github.com/prefix-dev/pixi) for more information on how to use it.

If `pixi` is not on Path, it can be installed with this command on Windows:

```pwsh
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

and on linux/MacOS:

```
curl -fsSL https://pixi.sh/install.sh | sh
```

To run Python in a virtual environment with the development package installed use the command: `pixi run python`
This command will start a Python interpreter with the development dependencies installed in a virtual environment.
Any additional arguments passed to this command will be passed to the Python interpreter in the virtual environment.

IMPORTANT! There is no python on the system path, so you must prefix any command with `pixi run <command>` to run in the main/test environment.
IMPORTANT! There is no python on the system path, so you must prefix any command with `pixi run -e lint` to run in the lint environment.

To run the tests use the command: `pixi run test`. This command will run `pytest` with the default arguments specified in the `pixi.toml` file.
Any extra arguments can be appended to the end of this command, for example, the name of a specific test, module or directory to run tests for.

To run the linter use the command: `pixi run pre-commit-run`.
This command runs `pre-commit` which in turns runs the configured linters and formatters on the codebase.
Any extra arguments appended to this command will be appended to the `pre-commit run` command.
