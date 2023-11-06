import sys

from qualtran_dev_tools.clean_notebooks import clean_notebooks
from qualtran_dev_tools.git_tools import get_git_root


def main():
    """Find, and strip metadata from all checked-in ipynbs."""
    reporoot = get_git_root()
    sourceroot = reporoot / 'qualtran'
    n_bad = clean_notebooks(sourceroot)
    sys.exit(n_bad)


if __name__ == '__main__':
    main()
