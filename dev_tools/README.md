# Development tooling

This directory contains scripts, modules, and configuration data used by developers.

`conf/` contains configuration files (for typecheckers, formatters, ...)

`requirements/` contains documentation and scripts for managing dependencies with `uv`.
End-user install requirements live in `pyproject.toml` at the repo root; pinned dev/CI
environments are defined in `uv.lock`. See `requirements/README` for install instructions.

`templates/` is used by the reference doc generation script.

At the top level of `dev_tools/.`, there are scripts that do various things. These scripts
should be run from the command line (if you know what you're doing). 

`qualtran_dev_tools/` contains reusable library code that may be helpful for writing 
developer-oriented scripts that e.g. do meta-analysis on the codebase. If you do
`pip install -e /path/to/Qualtran/dev_tools/`, it will install a package called 
`qualtran-dev-tools` which can be accessed from Python by `import qualtran_dev_tools.submodule`.
This package need not be installed if you're just interested in writing or analyzing quantum
algorithms and is really only useful for doing framework-scale meta-analysis.