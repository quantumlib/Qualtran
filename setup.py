import re

from setuptools import find_packages, setup


def version_number(path: str) -> str:
    """Get the version number from the src directory"""
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, "r") as f:
        version = version_re.search(f.read()).group(1)

    return version


def main() -> None:
    """ """
    version_path = "qualtran/_version.py"

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError("Version information not found in " + version_path)

    with open("README.md") as f:
        long_description = f.read()

    requirements = [
        r.strip()
        for r in open("dev_tools/requirements/deps/runtime.txt").readlines()
        if not r.startswith('#')
    ]

    setup(
        name="Qualtran",
        version=__version__,
        author="Google Quantum AI",
        author_email="mpharrigan@google.com",
        description="Software for fault-tolerant quantum algorithms research.",
        long_description=long_description,
        install_requires=requirements,
        license="Apache 2",
        packages=find_packages(),
    )


main()
