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
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List


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
    jq_code = '\n'.join(
        [
            '(.cells[] | select(has("outputs")) | .outputs) = []',
            '| (.cells[] | select(has("execution_count")) | .execution_count) = null',
            '| .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}',
        ]
    )
    cmd = ['jq', '--indent', '1', jq_code, nb_path]
    cp = subprocess.run(cmd, capture_output=True, universal_newlines=True, check=True)

    with NamedTemporaryFile('w', delete=False) as f:
        f.write(cp.stdout)

    cp = subprocess.run(['diff', nb_path, f.name], capture_output=True)
    dirty = len(cp.stdout) > 0
    print(str(nb_path))
    if dirty:
        print(cp.stdout.decode())
    if dirty and do_clean:
        os.rename(f.name, nb_path)

    return dirty


def clean_notebooks(sourceroot: Path) -> int:
    """Find, and strip metadata from all checked-in ipynbs."""
    nb_rel_paths = get_nb_rel_paths(rootdir=sourceroot)
    bad_nbs = []
    for nb_rel_path in nb_rel_paths:
        nbpath = sourceroot / nb_rel_path
        dirty = clean_notebook(nbpath)
        if dirty:
            bad_nbs.append(nb_rel_path)

    if len(bad_nbs) == 0:
        return 0

    print("Dirty notebooks: ")
    for nb_rel_path in bad_nbs:
        print(' ', str(nb_rel_path))

    return len(bad_nbs)
