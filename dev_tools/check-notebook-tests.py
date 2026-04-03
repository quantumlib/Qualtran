#  Copyright 2026 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Verify that every committed .ipynb has a corresponding pytest notebook test.

For each committed notebook, checks that:
1. A *_test.py file exists containing an execute_notebook('name') call
2. That test function is decorated with @pytest.mark.notebook

Usage:
    python dev_tools/check-notebook-tests.py
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

from qualtran_dev_tools.git_tools import get_git_root

_EXCLUDED_DIRS = {'dev_tools'}


def get_committed_notebooks(reporoot: Path) -> Dict[str, Path]:
    """Return {stem: relative_path} for all committed .ipynb files under reporoot.

    Excludes notebooks in dev_tools/ since those are developer utilities,
    not user-facing documentation.
    """
    result = subprocess.run(
        ['git', 'ls-files', '*.ipynb'], capture_output=True, text=True, check=True, cwd=reporoot
    )
    return {
        Path(f).stem: Path(f)
        for f in result.stdout.strip().split('\n')
        if f and not any(Path(f).parts[0] == d for d in _EXCLUDED_DIRS)
    }


def _is_notebook_marker(decorator: ast.expr) -> bool:
    """Check if a decorator is @pytest.mark.notebook."""
    # Handle pytest.mark.notebook (attr chain)
    if isinstance(decorator, ast.Attribute) and decorator.attr == 'notebook':
        return True
    return False


def find_notebook_tests(reporoot: Path) -> Dict[str, Tuple[Path, bool]]:
    """Find all execute_notebook() calls in test files.

    Searches all *_test.py files under the repo root (including qualtran/
    and tutorials/).

    Returns {notebook_name: (test_file_path_relative, has_notebook_marker)}.
    """
    results: Dict[str, Tuple[Path, bool]] = {}
    for test_file in reporoot.rglob('*_test.py'):
        try:
            tree = ast.parse(test_file.read_text())
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Check if function body contains execute_notebook('xxx')
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and _is_execute_notebook_call(child)
                    and child.args
                    and isinstance(child.args[0], ast.Constant)
                ):
                    nb_name = child.args[0].value
                    # Check for @pytest.mark.notebook decorator
                    has_marker = any(_is_notebook_marker(dec) for dec in node.decorator_list)
                    results[nb_name] = (test_file.relative_to(reporoot), has_marker)
    return results


def _is_execute_notebook_call(node: ast.Call) -> bool:
    """Check if a Call node is a call to execute_notebook (with or without module prefix)."""
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'execute_notebook':
        return True
    if isinstance(node.func, ast.Name) and node.func.id == 'execute_notebook':
        return True
    return False


def main():
    reporoot = get_git_root()

    committed = get_committed_notebooks(reporoot)
    tested = find_notebook_tests(reporoot)

    errors = []

    for stem, nb_rel_path in sorted(committed.items()):
        if stem not in tested:
            # Suggest the likely test file location
            nb_dir = nb_rel_path.parent
            test_file = nb_dir / f'{stem}_test.py'
            errors.append(
                f"  MISSING TEST: {nb_rel_path}\n"
                f"    Add to {test_file}:\n"
                f"\n"
                f"      @pytest.mark.notebook\n"
                f"      def test_{stem}_notebook():\n"
                f"          qlt_testing.execute_notebook('{stem}')\n"
            )
        else:
            test_file, has_marker = tested[stem]
            if not has_marker:
                errors.append(
                    f"  MISSING MARKER: {nb_rel_path}\n"
                    f"    The test in {test_file} calls execute_notebook('{stem}')\n"
                    f"    but is not decorated with @pytest.mark.notebook.\n"
                    f"    Add the decorator so this test runs in the notebooks CI job:\n"
                    f"\n"
                    f"      @pytest.mark.notebook\n"
                    f"      def test_...():\n"
                )

    if errors:
        print(f"ERROR: {len(errors)} notebook(s) have issues:\n")
        print("\n".join(errors))
        sys.exit(1)

    print(f"OK: All {len(committed)} notebooks have properly marked tests.")


if __name__ == '__main__':
    main()
