import os
import sys

from qualtran_dev_tools import shell_tools
from qualtran_dev_tools.incremental_coverage import check_for_uncovered_lines
from qualtran_dev_tools.prepared_env import PreparedEnv


def main():
    if len(sys.argv) < 2:
        print(
            shell_tools.highlight(
                'Must specify a comparison branch (e.g. "origin/master" or "HEAD~1").',
                shell_tools.RED,
            )
        )
        sys.exit(1)
    comparison_branch = sys.argv[1]

    env = PreparedEnv(
        actual_commit_id=None,  # local uncommitted files
        compare_commit_id=comparison_branch,
        destination_directory=os.getcwd(),
    )

    uncovered_count = check_for_uncovered_lines(env)
    if uncovered_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
