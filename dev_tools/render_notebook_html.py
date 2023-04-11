import argparse
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from cirq_qubitization.jupyter_tools import export_notebook

SOURCE_DIR_NAME = 'cirq_qubitization'
DOC_OUT_DIR_NAME = 'docs/nbs'


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
    fn = Path(DOC_OUT_DIR_NAME) / nb_rel_path.with_name(f'{nb_rel_path.stem}.html')
    print(fn)
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


def main(render_all: bool = False):
    """Find, execute, and export all checked-in ipynbs.

    Args:
        render_all: If False (the default) only render out-of-date html outputs.
    """
    render_all = True
    print("render_all:", render_all)
    reporoot = get_git_root()
    sourceroot = reporoot / SOURCE_DIR_NAME
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    nb_rel_paths = [p for p in nb_rel_paths if 'Tutorial' in str(p)]

    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        htmlpath = reporoot / get_html_rel_path(nb_rel_path)
        nbpath = sourceroot / nb_rel_path
        if not render_all and not is_out_of_date(nbpath, htmlpath):
            continue

        htmlpath.parent.mkdir(parents=True, exist_ok=True)
        err = export_notebook(nbpath=nbpath, htmlpath=htmlpath)
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
    p.add_argument('--all', action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    main(render_all=args.all)


if __name__ == '__main__':
    parse_args()
