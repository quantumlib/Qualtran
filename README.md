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
