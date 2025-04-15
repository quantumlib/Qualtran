<!-- # Qualtran (title omitted because our logo acts as the title) -->

<div align="center">
<img alt="Qualtran logo" width="340px" src="https://raw.githubusercontent.com/quantumlib/Qualtran/refs/heads/main/docs/_static/qualtran-logo-mode-sensitive.svg">
<br>

Python package for fault-tolerant quantum algorithms research.

[![Licensed under the Apache 2.0 open-source license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative\&logoColor=white\&style=flat-square)](https://github.com/quantumlib/qualtran/blob/main/LICENSE)
[![Compatible with Python versions 3.10 and higher](https://img.shields.io/badge/Python-3.10+-6828b2.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Qualtran project on PyPI](https://img.shields.io/pypi/v/qualtran.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=9d3bb8)](https://pypi.org/project/qualtran)

[Installation](#installation) &ndash;
[Usage](#usage) &ndash;
[Documentation](#documentation) &ndash;
[Community](#community) &ndash;
[Citation](#citation) &ndash;
[Contact](#contact)

</div>

Qualtran is a set of abstractions for representing quantum programs and a library of quantum
algorithms expressed in that language to support quantum algorithms research.

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

You should be able to import the `qualtran` package into your interactive Python environment as
as well as your programs:

```shell
import qualtran
```

If this is successful, you can move on to learning how to
[write bloqs](https://qualtran.readthedocs.io/en/latest/_infra/Bloqs-Tutorial.html) or investigate
the [bloqs library](https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library).

## Documentation

Documentation is available at https://qualtran.readthedocs.io/.

## Community

Qualtran's community is growing rapidly, and if you'd like to join the [many open-source
contributors] to the Qualtran project, we welcome your participation! We are dedicated to
cultivating an open and inclusive community, and have a [code of conduct].

[many open-source contributors]: https://github.com/quantumlib/Qualtran/graphs/contributors
[code of conduct]: https://github.com/quantumlib/Qualtran/blob/main/CODE_OF_CONDUCT.md

### Announcements

You can stay on top of Qualtran news using the approach that best suits your needs:

*   For releases and major announcements: join the low-volume mailing list [`qualtran-announce`].
*   For releases only:
    *   *Via GitHub notifications*: configure [repository notifications] for Qualtran.
    *   *Via RSS from GitHub*: subscribe to the GitHub [Qualtran releases feed].
    *   *Via RSS from PyPI*: subscribe to the [PyPI releases feed] for Qualtran.

[`qualtran-announce`]: https://groups.google.com/g/qualtran-announce
[repository notifications]: https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications
[Qualtran releases feed]: https://github.com/quantumlib/Qualtran/releases.atom
[PyPI releases feed]: https://pypi.org/rss/project/qualtran/releases.xml

### Questions and Discussions

*   If you'd like to ask questions and participate in discussions, join the [`qualtran-dev`]
    group/mailing list. By joining [`qualtran-dev`], you will also get automated invites to the
    biweekly _Qualtran Sync_ meeting (below).

*   Would you like to get more involved in Qualtran development? The biweekly _Qualtran Sync_
    is a virtual face-to-face meeting of contributors to discuss everything from issues to
    ongoing efforts, as well as to ask questions. Become a member of [`qualtran-dev`] to get
    an automatic meeting invitation!

[`qualtran-dev`]: https://groups.google.com/g/qualtran-dev

### Issues and Pull Requests

*   Do you have a feature request or want to report a bug? [Open an issue on
    GitHub] to report it!
*   Do you have a code contribution? Read our [contribution guidelines], then
    open a [pull request]!

[Open an issue on GitHub]: https://github.com/quantumlib/Qualtran/issues/new/choose
[contribution guidelines]: https://github.com/quantumlib/Qualtran/blob/main/CONTRIBUTING.md
[pull request]: https://help.github.com/articles/about-pull-requests

## Citation<a name="how-to-cite"></a>

When publishing articles or otherwise writing about Qualtran, please cite the following:

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

For any questions or concerns not addressed here, please email quantum-oss-maintainers@google.com.

## Disclaimer

This is not an officially supported Google product.
This project is not eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

Copyright 2025 Google LLC.

<div align="center">
  <a href="https://quantumai.google">
    <img width="15%" alt="Google Quantum AI"
         src="https://raw.githubusercontent.com/quantumlib/Qualtran/refs/heads/main/docs/_static/quantum-ai-vertical.svg">
  </a>
</div>
