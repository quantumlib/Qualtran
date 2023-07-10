.. _bloq_infra:

Infra
=====

``qualtran.bloq_infra`` contains the abstractions and infrastructure for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types (``Register``), and algorithms (``CompositeBloq``).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   components/Bloqs-Tutorial.ipynb
   components/composite_bloq.ipynb
   simulation/classical_sim.ipynb
   cirq_interop/cirq_conversion.ipynb
   drawing/graphviz.ipynb
   drawing/musical_score.ipynb
   resource_counting/bloq_counts.ipynb