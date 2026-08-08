---
trigger: always_on
description: "Enforce using uv for Python environment management, package installation, and running commands."
---

# Python Environment Guidelines

* Always use `uv` for managing Python virtual environments and dependencies.
* Install the locked environment with `uv sync --frozen [--no-dev] [--group ...]
  --no-install-project`, then `uv pip install --no-deps -e .` so dependencies come only
  from `uv.lock`. See `dev_tools/requirements/README` for details.
* Run Python commands, test suites, and linters via `uv run <command>` (e.g., `uv run check/pytest-quick`)