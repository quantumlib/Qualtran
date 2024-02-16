#  Copyright 2023 Google LLC
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
import asyncio
import multiprocessing
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import filelock
import nbconvert
import nbformat
from attrs import frozen
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode

from .git_tools import get_git_root


def get_nb_rel_paths(sourceroot: Path) -> List[Path]:
    """List all checked-in *.ipynb files within `sourceroot`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'],
        capture_output=True,
        universal_newlines=True,
        cwd=sourceroot,
        check=True,
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [Path(out) for out in outs]
    print(nb_rel_paths)
    return nb_rel_paths


def is_out_of_date(source_path: Path, out_path: Path) -> bool:
    """Whether the out_path file is stale according to its mtime."""
    assert source_path.exists()
    if not out_path.exists():
        return True

    source_mtime = source_path.stat().st_mtime
    output_mtime = out_path.stat().st_mtime
    return source_mtime > output_mtime


@frozen
class _NBInOutPaths:
    """A collection of input and output paths.

    If output paths are `None`, the requested file should not be
    output.
    """

    nb_in: Path
    html_out: Optional[Path]
    nb_out: Optional[Path]

    @classmethod
    def from_nb_rel_path(
        cls, nb_rel_path: Path, reporoot: Path, output_html: bool, output_nbs: bool
    ):
        sourceroot = reporoot / 'qualtran'
        nbpath = sourceroot / nb_rel_path

        html_rel_path = nb_rel_path.with_name(f'{nb_rel_path.stem}.html')
        html_outpath = reporoot / 'docs/nbrun' / html_rel_path if output_html else None
        nb_outpath = reporoot / 'docs' / nb_rel_path if output_nbs else None
        return cls(nb_in=nbpath, html_out=html_outpath, nb_out=nb_outpath)

    def html_needs_reexport(self) -> bool:
        """Whether the html output needs to be re-exported"""
        if self.html_out is None:
            return False

        return is_out_of_date(self.nb_in, self.html_out)

    def nb_needs_reexport(self) -> bool:
        """Whether the notebook output needs to be re-exported"""
        if self.nb_out is None:
            return False

        return is_out_of_date(self.nb_in, self.nb_out)

    def needs_reexport(self):
        """Whether anything needs to be re-exported"""
        return self.html_needs_reexport() or self.nb_needs_reexport()


def _make_link_replacements() -> List[Tuple[str, str]]:
    """Helper function to make a list of link replacements."""
    top_level = [
        'Bloq',
        'CompositeBloq',
        'BloqBuilder',
        'Register',
        'Signature',
        'Side',
        'BloqInstance',
        'Connection',
        'Soquet',
    ]
    replacements = [
        (f'`{name}`', f'[`{name}`](/reference/qualtran/{name}.md)') for name in top_level
    ]
    return replacements


_LINK_REPLACEMENTS = _make_link_replacements()


def linkify(nb: NotebookNode):
    """Replace certain object names with links to their reference docs in markdown cells."""
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'markdown':
            continue

        for fr_text, to_text in _LINK_REPLACEMENTS:
            cell.source = cell.source.replace(fr_text, to_text)


def execute_and_export_notebook(paths: _NBInOutPaths) -> Optional[Exception]:
    """Execute the notebook and export it in various forms.

    We execute the notebook at `nb_in` and export it as html to `html_out` and as
    a filled-in notebook to `nb_out` if the paths are not None.

    During execution, we catch any exceptions raised by notebook execution
    and return it. Otherwise, we return `None`.
    """

    # Load it in
    with paths.nb_in.open() as f:
        nb = nbformat.read(f, as_version=4)

    # Put in some cross-links. The following mutates `nb`
    linkify(nb)

    # Run it
    executor = ExecutePreprocessor(timeout=600, kernel_name="python3", record_timing=False)
    # Must manually lock the creation of the jupyter client to avoid a race condition.
    # https://github.com/jupyter/jupyter_client/issues/487#issuecomment-956206611
    executor.km = executor.create_kernel_manager()
    with filelock.FileLock("jupyter_kernel.lock", timeout=300):
        asyncio.run(executor.async_start_new_kernel())
        asyncio.run(executor.async_start_new_kernel_client())

    print(f"Executing {paths.nb_in.stem}")
    try:
        nb, resources = executor.preprocess(nb)
    except Exception as e:  # pylint: disable=broad-except
        print(f'{paths.nb_in} failed!')
        print(e)
        return e

    # Optionally save executed notebook as notebook
    if paths.nb_out:
        paths.nb_out.parent.mkdir(parents=True, exist_ok=True)
        with paths.nb_out.open('w') as f:
            nbformat.write(nb, f, version=4)

    # Optionally save as html
    if paths.html_out:
        html, resources = nbconvert.export(nbconvert.HTMLExporter(), nb, resources=resources)
        paths.html_out.parent.mkdir(parents=True, exist_ok=True)
        with paths.html_out.open('w') as f:
            f.write(html)


class _NotebookRunClosure:
    """Used to run notebook execution logic in subprocesses."""

    def __init__(self, reporoot: Path, output_nbs: bool, output_html: bool, only_out_of_date: bool):
        self.reporoot = reporoot
        self.output_nbs = output_nbs
        self.output_html = output_html
        self.only_out_of_date = only_out_of_date

    def __call__(self, nb_rel_path: Path) -> Tuple[Path, Optional[Exception]]:
        paths = _NBInOutPaths.from_nb_rel_path(
            nb_rel_path, self.reporoot, output_html=self.output_html, output_nbs=self.output_nbs
        )

        if self.only_out_of_date and not paths.needs_reexport():
            print(f'{nb_rel_path} up to date')
            return paths.nb_in, None

        err = execute_and_export_notebook(paths)
        print(f"Exported {nb_rel_path}")
        return paths.nb_in, err


def execute_and_export_notebooks(
    output_nbs: bool, output_html: bool, only_out_of_date: bool = True
):
    """Find, execute, and export all checked-in ipynbs.

    Args:
        output_nbs: Whether to save the executed notebooks as notebooks
        output_html: Whether to save the executed notebooks as html
        only_out_of_date: Only re-execute and re-export notebooks whose output files
            are out of date.
    """
    reporoot = get_git_root()
    sourceroot = reporoot / 'qualtran'
    nb_rel_paths = get_nb_rel_paths(sourceroot=sourceroot)
    func = _NotebookRunClosure(
        reporoot=reporoot,
        output_nbs=output_nbs,
        output_html=output_html,
        only_out_of_date=only_out_of_date,
    )
    with multiprocessing.Pool() as pool:
        results = pool.map(func, nb_rel_paths)
    bad_nbs = [nbname for nbname, err in results if err is not None]

    if len(bad_nbs) > 0:
        print()
        print("Errors in notebooks:")
        for nb in bad_nbs:
            print(' ', nb)
        sys.exit(1)
