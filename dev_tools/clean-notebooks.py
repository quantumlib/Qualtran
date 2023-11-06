import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

from qualtran_dev_tools.git_tools import get_git_root
from qualtran_dev_tools.clean_notebooks import clean_notebooks

def main():
    """Find, and strip metadata from all checked-in ipynbs."""
    reporoot = get_git_root()
    sourceroot = reporoot / 'qualtran'
    n_bad = clean_notebooks(sourceroot)
    sys.exit(n_bad)

if __name__ == '__main__':
    main()
