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
