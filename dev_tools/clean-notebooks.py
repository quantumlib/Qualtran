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
import argparse
import sys

from qualtran_dev_tools.clean_notebooks import clean_notebooks
from qualtran_dev_tools.git_tools import get_git_root


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    reporoot = get_git_root()
    sourceroot = reporoot / 'qualtran'
    n_bad = clean_notebooks(sourceroot, do_clean=args.apply)
    sys.exit(n_bad)


if __name__ == '__main__':
    parse_args()
