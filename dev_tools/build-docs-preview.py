#!/usr/bin/env python3
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

"""Script to automate local preview build of Qualtran documentation."""

import argparse
import pathlib
import subprocess
import sys
import webbrowser


def run_command(cmd, cwd):
    print(f"Running: {' '.join(cmd)} in {cwd}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Automate local preview build of Qualtran documentation."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Perform a clean build (rebuilds everything from scratch).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated documentation in a web browser.",
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    docs_dir = repo_root / "docs"

    print(f"Script directory: {script_dir}")
    print(f"Repo root: {repo_root}")
    print(f"Docs directory: {docs_dir}")

    make_cmd = "make.bat" if sys.platform == "win32" else "make"

    if args.clean:
        print("Starting clean build...")
        # 1. Run git clean -fdX in the docs/ directory
        run_command(["git", "clean", "-fdX"], cwd=docs_dir)

        # 2. Run python execute-notebooks.py --no-only-out-of-date in dev_tools/
        run_command(
            [sys.executable, "execute-notebooks.py", "--no-only-out-of-date"], cwd=script_dir
        )

        # 3. Run python build-reference-docs-2.py in dev_tools/
        run_command([sys.executable, "build-reference-docs-2.py"], cwd=script_dir)

        # 4. Run make clean and make html in docs/
        run_command([make_cmd, "clean"], cwd=docs_dir)
        run_command([make_cmd, "html"], cwd=docs_dir)
    else:
        print("Starting default build...")
        # 1. Skip cleaning the docs/ directory

        # 2. Run python execute-notebooks.py in dev_tools/
        run_command([sys.executable, "execute-notebooks.py"], cwd=script_dir)

        # 3. Run python build-reference-docs-2.py in dev_tools/
        run_command([sys.executable, "build-reference-docs-2.py"], cwd=script_dir)

        # 4. Run make html in docs/
        run_command([make_cmd, "html"], cwd=docs_dir)

    print("Doc build completed successfully.")

    if not args.no_open:
        index_file = docs_dir / "_build" / "html" / "index.html"
        if index_file.exists():
            url = index_file.resolve().as_uri()
            print(f"Opening documentation in browser: {url}")
            try:
                if not webbrowser.open(url):
                    print("Warning: Failed to open browser.", file=sys.stderr)
            except Exception as e:
                print(f"Failed to open browser: {e}", file=sys.stderr)
        else:
            print(f"Warning: Documentation index file not found at {index_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
