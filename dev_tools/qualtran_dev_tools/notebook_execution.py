#  Copyright 2023 Google Quantum AI
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

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import nbconvert
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from .git_tools import get_git_root


def get_nb_rel_paths(rootdir) -> List[Path]:
    """List all checked-in *.ipynb files within `rootdir`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'], capture_output=True, universal_newlines=True, cwd=rootdir
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [Path(out) for out in outs]
    print(nb_rel_paths)
    return nb_rel_paths


def is_out_of_date(source_path: Path, out_path: Path) -> bool:
    """Whether the out_path file needs to be re-rendered according to its mtime."""
    assert source_path.exists()
    if not out_path.exists():
        print(' -> Out of date: output file does not exist')
        return True

    source_mtime = source_path.stat().st_mtime
    output_mtime = out_path.stat().st_mtime
    ood = source_mtime > output_mtime
    if ood:
        print(f' -> Out of date: {source_mtime} > {output_mtime}')
    else:
        print(' -> Up to date.')
    return ood


def execute_and_export_notebook(
    nbpath: Path, html_outpath: Optional[Path], nb_outpath: Optional[Path]
) -> Optional[Exception]:
    """Execute the notebook and export it in various forms.

    We execute the notebook at `nbpath` and export it as html to `html_outpath` and as
    a filled-in notebook to `nb_outpath` if the paths are not None.

    During execution, we catch any exceptions raised by notebook execution
    and return it. Otherwise, we return `None`.
    """

    # Load it in
    with nbpath.open() as f:
        nb = nbformat.read(f, as_version=4)

    # Run it
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        nb, resources = ep.preprocess(nb)
    except Exception as e:
        print(f'{nbpath} failed!')
        print(e)
        return e

    # Optionally save executed notebook as notebook
    if nb_outpath:
        nb_outpath.parent.mkdir(parents=True, exist_ok=True)
        with nb_outpath.open('w') as f:
            nbformat.write(nb, f, version=4)

    # Optionally save as html
    if html_outpath:
        html, resources = nbconvert.export(nbconvert.HTMLExporter(), nb, resources=resources)
        html_outpath.parent.mkdir(parents=True, exist_ok=True)
        with html_outpath.open('w') as f:
            f.write(html)


def execute_and_export_notebooks(output_nbs: bool, output_html: bool):
    """Find, execute, and export all checked-in ipynbs.

    Args:
        output_nbs: Whether to save the executed notebooks as notebooks
        output_html: Whether to save the executed notebooks as html
    """
    reporoot = get_git_root()
    sourceroot = reporoot / 'qualtran'
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        print(f"Executing {nb_rel_path}")
        nbpath = sourceroot / nb_rel_path

        html_rel_path = nb_rel_path.with_name(f'{nb_rel_path.stem}.html')
        html_outpath = reporoot / 'docs/nbrun' / html_rel_path if output_html else None
        nb_outpath = reporoot / 'docs' / nb_rel_path if output_nbs else None

        err = execute_and_export_notebook(
            nbpath=nbpath, html_outpath=html_outpath, nb_outpath=nb_outpath
        )
        if err is not None:
            bad_nbs.append(nbpath)

    if len(bad_nbs) > 0:
        print()
        print("Errors in notebooks:")
        for nb in bad_nbs:
            print(' ', nb)
        sys.exit(1)
