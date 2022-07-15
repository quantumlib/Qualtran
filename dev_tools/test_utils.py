import os

import pytest


def only_on_posix(func):
    """Only run test on posix."""
    return pytest.mark.skipif(os.name != "posix", reason=f"os {os.name} is not posix")(
        func
    )
