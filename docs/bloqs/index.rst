.. _bloqs_library:

Bloqs Library
=============

``qualtran.bloqs`` contains implementations of quantum operations and subroutines.

.. todo: Make module organization match page organization
    and_bloq -> basic_gates
    swap_network -> basic_gates?
    ?? -> mean estimation
    everything in "Other" -> ??

.. todo: generalized qsp needs a title.

.. todo: swap_network_2 should be removed after jupyter notebook autogen migration.

.. toctree::
    :maxdepth: 2
    :caption: Basic Gates:

    basic_gates.ipynb
    and_bloq.ipynb
    swap_network.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Chemistry:

    chemistry/sparse/sparse.ipynb
    chemistry/thc/thc.ipynb
    chemistry/trotter/trotter.ipynb
    chemistry/pbc/first_quantization/first_quantization.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Factoring:

    factoring/factoring-via-modexp.ipynb
    factoring/ref-factoring.ipynb
    factoring/mod_exp.ipynb
    factoring/mod_mul.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Arithmetic:

    arithmetic/arithmetic.ipynb
    arithmetic/comparison_gates.ipynb

.. toctree::
    :maxdepth: 2
    :caption: Other:

    apply_gate_to_lth_target.ipynb
    hubbard_model.ipynb
    phase_estimation_of_quantum_walk.ipynb
    prepare_uniform_superposition.ipynb
    qrom.ipynb
    qubitization_walk_operator.ipynb
    select_pauli_lcu.ipynb
    sorting.ipynb
    state_preparation.ipynb
    unary_iteration.ipynb
    util_bloqs.ipynb
    generalized_qsp.ipynb
