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

"""Clean the files outputted by "stage 2" doc generation. See `how-to-build-the-docs.md` for
what this means.

If you're on the `main` branch this script is unnecessary as you can use `git clean -fdx`
from the `docs/` directory to achieve the same result. On the `main` branch, the generated
outputs are ignored.

If you're on the `docs` branch, this script is necessary.
"""

import shutil

from qualtran_dev_tools.git_tools import get_git_root


def main():
    docs = get_git_root() / 'docs'

    checkpoint_dirs = docs.glob('**/.ipynb_checkpoints')
    for checkpoint_dir in checkpoint_dirs:
        assert checkpoint_dir.is_dir(), checkpoint_dir
        print("Removing", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)

    mds = docs.glob('**/*.md')
    for md in mds:
        print("Removing", md)
        md.unlink()

    ipynbs = docs.glob('**/*.ipynb')
    for nb in ipynbs:
        print("Removing", nb)
        nb.unlink()

    autotoc = docs / 'reference/autotoc.rst'
    if autotoc.exists():
        print("Removing", autotoc)
        autotoc.unlink()


if __name__ == '__main__':
    main()
