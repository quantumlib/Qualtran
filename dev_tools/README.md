# Development tooling

This directory contains scripts, modules, and configuration data used by developers.

`conf/` contains configuration files (for typecheckers, formatters, ...)

`requirements/` contain the sources-of-truth for the various dependencies of the project.
Notably, `requirements/deps/runtime.txt` contains the requirements that `pip install qualtran`
will try to satisfy. The `requirements/` directory contains additional developer dependencies
as well as pinned-version environment specifications.

`templates/` is used by the reference doc generation script.

At the top level of `dev_tools/.`, there are scripts that do various things. These scripts
should be run from the command line (if you know what you're doing). 

`qualtran_dev_tools/` contains re-usable library code that may be helpful for writing 
developer-oriented scripts that e.g. do meta-analysis on the codebase. If you do
`pip install -e /path/to/Qualtran/dev_tools/`, it will install a package called 
`qualtran-dev-tools` which can be accessed from Python by `import qualtran_dev_tools.submodule`.
This package need not be installed if you're just interested in writing or analyzing quantum
algorithms and is really only useful for doing framework-scale meta-analysis.