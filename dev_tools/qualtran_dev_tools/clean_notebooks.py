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
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import nbformat
from nbconvert.preprocessors import ClearMetadataPreprocessor


def get_nb_rel_paths(rootdir) -> List[Path]:
    """List all checked-in *.ipynb files within `rootdir`."""
    cp = subprocess.run(
        ['git', 'ls-files', '*.ipynb'], capture_output=True, universal_newlines=True, cwd=rootdir
    )
    outs = cp.stdout.splitlines()
    nb_rel_paths = [Path(out) for out in outs]
    print(nb_rel_paths)
    return nb_rel_paths


def clean_notebook(nb_path: Path, do_clean: bool = True):
    """Clean a notebook.

    If `do_clean` is true, modify the notebook. Otherwise, just print a diff.
    """
    with nb_path.open() as f:
        nb = nbformat.read(f, as_version=4)

    pp = ClearMetadataPreprocessor(preserve_cell_metadata_mask={'cq.autogen'})
    nb, resources = pp.preprocess(nb, resources={})

    with NamedTemporaryFile('w', delete=False) as f:
        nbformat.write(nb, f, version=4)

    res = subprocess.run(['diff', nb_path, f.name], capture_output=True)
    dirty = len(res.stdout) > 0
    print(str(nb_path))
    if dirty:
        print(res.stdout.decode())
    if dirty and do_clean:
        os.rename(f.name, nb_path)
    if not do_clean:
        os.unlink(f.name)

    return dirty


def clean_notebooks(sourceroot: Path, do_clean: bool = True) -> int:
    """Find, and strip metadata from all checked-in ipynbs.

    If `do_clean` is true, modify the notebook. Otherwise, just print a diff.
    """
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        nbpath = sourceroot / nb_rel_path
        dirty = clean_notebook(nbpath, do_clean=do_clean)
        if dirty:
            bad_nbs.append(nb_rel_path)

    if len(bad_nbs) == 0:
        return 0

    print("Dirty notebooks: ")
    for nb_rel_path in bad_nbs:
        print(' ', str(nb_rel_path))

    return len(bad_nbs)
