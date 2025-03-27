<!-- H1 title omitted because our logo acts as the title. -->

<div align="center">
<img alt="Qualtran logo" width="340px" src="docs/_static/qualtran-logo-black.svg#gh-light-mode-only">
<img alt="Qualtran logo" width="340px" src="docs/_static/qualtran-logo-white.svg#gh-dark-mode-only">
<br>

Python package for fault-tolerant quantum algorithms research.

[![Licensed under the Apache 2.0 open-source license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative\&logoColor=white\&style=flat-square)](https://github.com/quantumlib/qualtran/blob/main/LICENSE)
[![Compatible with Python versions 3.10 and higher](https://img.shields.io/badge/Python-3.10+-6828b2.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Qualtran project on PyPI](https://img.shields.io/pypi/v/qualtran.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=9d3bb8)](https://pypi.org/project/qualtran)

[Installation](#installation) &ndash;
[Usage](#usage) &ndash;
[Documentation](#documentation) &ndash;
[News](#news) &ndash;
[Citation](#citation) &ndash;
[Contact](#contact)

</div>

Qualtran (_quantum algorithms translator_) is a set of abstractions for representing quantum
programs and a library of quantum algorithms expressed in that language to support quantum
algorithms research.

## Installation

Qualtran is being actively developed. We recommend installing from the source code.

The following commands will clone a copy of the repository, then install the Qualtran package in
your local Python environment as a local editable copy:

```shell
git clone https://github.com/quantumlib/Qualtran.git
cd Qualtran/
pip install -e .
```

You can also install the latest tagged release using `pip`:

```shell
pip install qualtran
```

You can also install the latest version of the main branch on GitHub:

```shell
pip install git+https://github.com/quantumlib/Qualtran
```

## Usage

> [!WARNING]
> Qualtran is an experimental preview release. We provide no backwards compatibility guarantees.
> Some algorithms or library functionality may be incomplete or contain inaccuracies. Open issues or
> contact the authors with bug reports or feedback.

### Python interpreter and programs

You should be able to import the `qualtran` package into your interactive Python environment as
as well as your programs:

```shell
import qualtran
```

If this is successful, you can move on to learning how to
[write bloqs](https://qualtran.readthedocs.io/en/latest/_infra/Bloqs-Tutorial.html) or investigate
the [bloqs library](https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library).

### Physical Resource Estimation GUI

Qualtran provides a GUI for estimating the physical resources (qubits, magic states, runtime, etc.)
needed to run a quantum algorithm. The GUI can be run locally by running:

```shell
cd $QUALTRAN_HOME
python -m qualtran.surface_code.ui
```

## Documentation

Documentation is available at https://qualtran.readthedocs.io/.

## News

Stay on top of Qualtran developments using the approach that best suits your needs:

*   For news and updates announcements: sign up to the low-volume mailing list
    [`qualtran-announce`].
*   For releases only:
    *   *Via GitHub notifications*: configure [repository notifications] for Qualtran.
    *   *Via RSS from GitHub*: subscribe to the GitHub [Qualtran releases feed].
    *   *Via RSS from PyPI*: subscribe to the [PyPI releases feed] for Qualtran.

[`qualtran-announce`]: https://groups.google.com/g/qualtran-announce
[repository notifications]: https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications
[Qualtran releases feed]: https://github.com/quantumlib/Qualtran/releases.atom
[PyPI releases feed]: https://pypi.org/rss/project/qualtran/releases.xml

## Citation<a name="how-to-cite"></a>

When publishing articles or otherwise writing about Qualtran, please cite the
following:

```bibtex
@misc{harrigan2024qualtran,
    title={Expressing and Analyzing Quantum Algorithms with Qualtran},
    author={Matthew P. Harrigan and Tanuj Khattar
        and Charles Yuan and Anurudh Peduri and Noureldin Yosri
        and Fionn D. Malone and Ryan Babbush and Nicholas C. Rubin},
    year={2024},
    eprint={2409.04643},
    archivePrefix={arXiv},
    primaryClass={quant-ph},
    doi={10.48550/arXiv.2409.04643},
    url={https://arxiv.org/abs/2409.04643},
}
```

## Contact

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="./docs/_static/quantum-ai-vertical.svg">
  </a>
</div>
