.. _bloq_infra:

Fundamentals
============

``qualtran`` and its submodules contain the abstractions and infrastructure for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types (``Register``), and algorithms (``CompositeBloq``).

.. toctree::
   :maxdepth: 1

   _infra/Bloqs-Tutorial.ipynb
   Protocols.ipynb
   simulation/classical_sim.ipynb
   simulation/tensor.ipynb
   resource_counting/bloq_counts.ipynb
   Adjoint.ipynb
   Controlled.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Quantum Computer Architectures:

   surface_code/azure_cost_model.ipynb
   surface_code/thc_compilation.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics:

   _infra/composite_bloq.ipynb
   cirq_interop/cirq_interop.ipynb
   cirq_interop/t_complexity.ipynb
   _infra/gate_with_registers.ipynb
   drawing/graphviz.ipynb
   drawing/musical_score.ipynb
