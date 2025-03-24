# Qᴜᴀʟᴛʀᴀɴ

Qᴜᴀʟᴛʀᴀɴ (quantum algorithms translator) is a set of abstractions for representing quantum
programs and a library of quantum algorithms expressed in that language to support quantum
algorithms research.

**Note:** Qualtran is an experimental preview release. We provide no backwards compatibility
guarantees. Some algorithms or library functionality may be incomplete or contain inaccuracies.
Open issues or contact the authors with bug reports or feedback.

Subscribe to [qualtran-announce@googlegroups.com](https://groups.google.com/g/qualtran-announce)
to receive the latest news and updates!

## Documentation

Documentation is available at https://qualtran.readthedocs.io/

## Installation

Qualtran is being actively developed. We recommend installing from source:

For a local editable copy:

    git clone https://github.com/quantumlib/Qualtran.git
    cd Qualtran/
    pip install -e .

You can also install the latest tagged release using `pip`:

    pip install qualtran

You can also install the latest state of the main branch:

    pip install git+https://github.com/quantumlib/Qualtran

## Physical Resource Estimation GUI

Qualtran provides a GUI for estimating the physical resources (qubits, magic states, runtime, ..etc) needed to run a quantum algorithm. The GUI can be run locally by running:

    cd $QUALTRAN_HOME
    python -m qualtran.surface_code.ui

## Citation

When publishing articles or otherwise writing about Qualtran, please cite the
following:

```latex
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
