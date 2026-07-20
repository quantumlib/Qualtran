---
trigger: always_on
description: "Enforce using uv for Python environment management, package installation, and running commands."
---

# Python Environment Guidelines

* Always use `uv` for managing Python virtual environments and dependencies.
* Run Python commands, test suites, and linters via `uv run <command>` (e.g., `uv run check/pytest-quick`