import argparse
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import nbconvert
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def get_git_root() -> Path:
    """Get the root git repository path."""
    cp = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], capture_output=True, universal_newlines=True
    )
    path = Path(cp.stdout.strip()).absolute()
    assert path.exists()
    print('git root', path)
    return path


def get_nb_rel_paths(rootdir) -> List[Path]:
    """List all checked-in *.ipynb files within `rootdir`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'], capture_output=True, universal_newlines=True, cwd=rootdir
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [Path(out) for out in outs]
    print(nb_rel_paths)
    return nb_rel_paths


def get_html_rel_path(nb_rel_path: Path) -> Path:
    """For a given input relative notebook path, get the desired output html path."""
    fn = nb_rel_path.with_name(f'{nb_rel_path.stem}.html')
    return fn


def is_out_of_date(nbpath: Path, htmlpath: Path) -> bool:
    """Whether the html file needs to be re-rendered according to its mtime."""
    assert nbpath.exists()
    if not htmlpath.exists():
        print(' -> Out of date: output file does not exist')
        return True

    nbmtime = nbpath.stat().st_mtime
    htmtime = htmlpath.stat().st_mtime
    ood = nbmtime > htmtime
    if ood:
        print(f' -> Out of date: {nbmtime} > {htmtime}')
    else:
        print(' -> Up to date.')
    return ood


def execute_and_export_nb(nbpath: Path, htmlpath: Path, nboutpath: Path) -> Optional[Exception]:
    """Execute the notebook and export it in various forms.

    We execute the notebook at `nbpath` and export it as html to `htmlpath` and as
    a filled-in notebook to `nboutpath`.

    During execution, we catch any exceptions raised by notebook execution
    and return it. Otherwise, we return `None`.
    """
    with nbpath.open() as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        nb, resources = ep.preprocess(nb)
    except Exception as e:
        print(f'{nbpath} failed!')
        print(e)
        return e

    with nboutpath.open('w') as f:
        nbformat.write(nb, f, version=4)
    html, resources = nbconvert.export(nbconvert.HTMLExporter(), nb, resources=resources)
    with htmlpath.open('w') as f:
        f.write(html)


def main(render_all: bool = False):
    """Find, execute, and export all checked-in ipynbs.

    Args:
        render_all: If False (the default) only render out-of-date html outputs.
    """
    print("render_all:", render_all)
    reporoot = get_git_root()
    sourceroot = reporoot / 'cirq_qubitization'
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        print(f"Executing {nb_rel_path}")
        htmlpath = reporoot / 'docs/nbrun' / get_html_rel_path(nb_rel_path)
        nbpath = sourceroot / nb_rel_path
        nboutpath = reporoot / 'docs' / nb_rel_path

        if not render_all and not is_out_of_date(nbpath, htmlpath):
            continue

        htmlpath.parent.mkdir(parents=True, exist_ok=True)
        nboutpath.parent.mkdir(parents=True, exist_ok=True)

        err = execute_and_export_nb(nbpath=nbpath, htmlpath=htmlpath, nboutpath=nboutpath)
        if err is not None:
            bad_nbs.append(nbpath)

    if len(bad_nbs) > 0:
        print()
        print("Errors in notebooks:")
        for nb in bad_nbs:
            print(' ', nb)
        sys.exit(1)


def parse_args():
    p = ArgumentParser()
    p.add_argument('--all', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    main(render_all=args.all)


if __name__ == '__main__':
    parse_args()
