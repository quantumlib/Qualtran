from typing import List, Optional

from . import shell_tools


class PreparedEnv:
    """Details of a local environment that has been prepared for use."""

    def __init__(
        self,
        actual_commit_id: Optional[str],
        compare_commit_id: str,
        destination_directory: Optional[str],
    ) -> None:
        """Initializes a description of a prepared (or desired) environment.

        Args:
            github_repo: The github repository that the local environment
                corresponds to. Use None if the actual_commit_id corresponds
                to a commit that isn't actually on github.
            actual_commit_id: Identifies the commit that has been checked out
                for testing purposes. Use None for 'local uncommitted changes'.
            compare_commit_id: Identifies a commit that the actual commit can
                be compared against, e.g. when diffing for incremental checks.
            destination_directory: The location where the environment has been
                prepared. If the directory isn't prepared yet, this should be
                None.
        """
        self.actual_commit_id = actual_commit_id
        self.compare_commit_id = compare_commit_id
        if self.compare_commit_id == self.actual_commit_id:
            self.compare_commit_id += "~1"

        self.destination_directory = destination_directory

    def get_changed_files(self) -> List[str]:
        """Get the files changed on one git branch vs another.

        Returns:
            List[str]: File paths of changed files, relative to the git repo
                root.
        """
        optional_actual_commit_id = [] if self.actual_commit_id is None else [self.actual_commit_id]
        out = shell_tools.output_of(
            [
                "git",
                "diff",
                "--name-only",
                self.compare_commit_id,
                *optional_actual_commit_id,
                "--",
            ],
            cwd=self.destination_directory,
        )
        return [e for e in out.split("\n") if e.strip()]
