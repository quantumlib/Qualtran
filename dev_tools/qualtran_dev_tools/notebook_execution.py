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
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import attrs
import filelock
import nbconvert
import nbformat
from attrs import frozen
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import NotebookNode

from .git_tools import get_git_root


def get_nb_rel_paths(sourceroot: Path) -> List[Tuple[Path, Path]]:
    """List all checked-in *.ipynb files within `sourceroot`.

    Returns a tuple of the relative path and the path it's relative to (aka sourceroot)
    """
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'],
        capture_output=True,
        universal_newlines=True,
        cwd=sourceroot,
        check=True,
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [(Path(out), sourceroot) for out in outs]
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
    md_out: Optional[Path]
    nb_out: Optional[Path]

    @classmethod
    def from_nb_rel_path(
        cls, nb_rel_path: Path, reporoot: Path, sourceroot: Path, output_md: bool, output_nbs: bool
    ):
        nbpath = sourceroot / nb_rel_path

        md_rel_path = nb_rel_path.with_name(f'{nb_rel_path.stem}.md')
        md_outpath = reporoot / 'docs' / md_rel_path if output_md else None
        nb_outpath = reporoot / 'docs' / nb_rel_path if output_nbs else None
        return cls(nb_in=nbpath, md_out=md_outpath, nb_out=nb_outpath)

    def md_needs_reexport(self) -> bool:
        """Whether the md output needs to be re-exported"""
        if self.md_out is None:
            return False

        return is_out_of_date(self.nb_in, self.md_out)

    def nb_needs_reexport(self) -> bool:
        """Whether the notebook output needs to be re-exported"""
        if self.nb_out is None:
            return False

        return is_out_of_date(self.nb_in, self.nb_out)

    def needs_reexport(self):
        """Whether anything needs to be re-exported"""
        return self.md_needs_reexport() or self.nb_needs_reexport()


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

    # Optionally save as markdown
    if paths.md_out:
        md, resources = nbconvert.export(nbconvert.MarkdownExporter(), nb, resources=resources)
        paths.md_out.parent.mkdir(parents=True, exist_ok=True)
        with paths.md_out.open('w') as f:
            f.write(md)

    return None


@attrs.frozen
class _NotebookRunResult:
    nb_in: Path
    err: Optional[Exception]
    duration_s: float


class _NotebookRunClosure:
    """Used to run notebook execution logic in subprocesses."""

    def __init__(self, reporoot: Path, output_nbs: bool, output_md: bool, only_out_of_date: bool):
        self.reporoot = reporoot
        self.output_nbs = output_nbs
        self.output_md = output_md
        self.only_out_of_date = only_out_of_date

    def __call__(self, nb_rel_path: Path, sourceroot: Path) -> _NotebookRunResult:
        paths = _NBInOutPaths.from_nb_rel_path(
            nb_rel_path,
            reporoot=self.reporoot,
            sourceroot=sourceroot,
            output_md=self.output_md,
            output_nbs=self.output_nbs,
        )

        if self.only_out_of_date and not paths.needs_reexport():
            print(f'{nb_rel_path} up to date')
            return _NotebookRunResult(paths.nb_in, None, 0.0)

        start = time.time()
        err = execute_and_export_notebook(paths)
        end = time.time()
        print(f"Exported {nb_rel_path} in {end-start:.2f} seconds.")
        return _NotebookRunResult(paths.nb_in, err, duration_s=end - start)


def execute_and_export_notebooks(
    *,
    output_nbs: bool,
    output_md: bool,
    only_out_of_date: bool = True,
    n_workers: Optional[int] = None,
):
    """Find, execute, and export all checked-in ipynbs.

    Args:
        output_nbs: Whether to save the executed notebooks as notebooks
        output_md: Whether to save the executed notebooks as markdown
        only_out_of_date: Only re-execute and re-export notebooks whose output files
            are out of date.
        n_workers: If set to 1, do not use parallelization. Otherwise, use
            `multiprocessing.Pool(n_workers)` to execute notebooks in parallel.
    """
    reporoot = get_git_root()
    nb_rel_paths = get_nb_rel_paths(sourceroot=reporoot / 'qualtran')
    nb_rel_paths += get_nb_rel_paths(sourceroot=reporoot / 'tutorials')
    random.shuffle(nb_rel_paths)
    print(f"Found {len(nb_rel_paths)} notebooks.")
    func = _NotebookRunClosure(
        reporoot=reporoot,
        output_nbs=output_nbs,
        output_md=output_md,
        only_out_of_date=only_out_of_date,
    )
    if n_workers == 1:
        print("(Not using multiprocessing, n_workers=1)")
        results = [func(nb_rel_path, sourceroot) for nb_rel_path, sourceroot in nb_rel_paths]
    else:
        print(f"Multiprocessing with {n_workers=}")
        with multiprocessing.Pool(n_workers, maxtasksperchild=1) as pool:
            results = pool.starmap(func, nb_rel_paths)
        assert results
    bad_nbs = [result.nb_in for result in results if result.err is not None]

    if len(bad_nbs) > 0:
        print()
        print("Errors in notebooks:")
        for nb in bad_nbs:
            print(' ', nb)
        sys.exit(1)

    duration_nbs = sorted(results, key=lambda r: r.duration_s, reverse=True)
    print("Slowest 10 notebooks:")
    for result in duration_nbs[:10]:
        print(f'{result.duration_s:5.2f}s {result.nb_in}')
