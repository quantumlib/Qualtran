import io
import re

from setuptools import find_packages, setup


def version_number(path: str) -> str:
    """Get cirq-qubitization's version number from the src directory"""
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, "r") as fqe_version:
        version = version_re.search(fqe_version.read()).group(1)

    return version


def main() -> None:
    """ """
    version_path = "qualtran/_version.py"

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError("Version information not found in " + version_path)

    long_description = "=================\n" + "CIRQ-QUBITIZATION\n" + "=================\n"
    stream = io.open("README.md", encoding="utf-8")
    stream.readline()
    long_description += stream.read()

    requirements = [
        r.strip()
        for r in open("dev_tools/requirements/deps/runtime.txt").readlines()
        if not r.startswith('#')
    ]

    setup(
        name="qualtran",
        version=__version__,
        author="Nicholas C. Rubin and Tanuj Khattar",
        author_email="rubinnc0@gmail.com",
        description="Learning tools and basics for quantum chemistry",
        long_description=long_description,
        install_requires=requirements,
        license="Apache 2",
        packages=find_packages(),
    )


main()
