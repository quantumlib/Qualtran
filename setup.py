"""
CIRQ-QUBITIZATION installation script
"""

import io
import re

from setuptools import setup, find_packages
from dev_tools.requirements import explode


def version_number(path: str) -> str:
    """Get cirq-qubitization's version number from the src directory"""
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, "r") as fqe_version:
        version = version_re.search(fqe_version.read()).group(1)

    return version


def main() -> None:
    """ """
    version_path = "cirq_qubitization/_version.py"

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError("Version information not found in " + version_path)

    long_description = (
        "=================\n" + "CIRQ-QUBITIZATION\n" + "=================\n"
    )
    stream = io.open("README.md", encoding="utf-8")
    stream.readline()
    long_description += stream.read()

    requirements = explode("dev_tools/requirements/deps/cirq-qubitization-all.txt")
    dev_requirements = explode("dev_tools/requirements/dev.env.txt")
    # requirements = [r.strip() for r in requirements_buffer]

    setup(
        name="cirq_qubitization",
        version=__version__,
        author="Nicholas C. Rubin and Tanuj Khattar",
        author_email="rubinnc0@gmail.com",
        description="Learning tools and basics for quantum chemistry",
        long_description=long_description,
        install_requires=requirements,
        extras_require={"dev_env": dev_requirements},
        license="Apache 2",
        packages=find_packages(),
    )


main()
