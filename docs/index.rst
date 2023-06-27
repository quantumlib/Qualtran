Qᴜᴀʟᴛʀᴀɴ
========

Quantum computing hardware continues to advance. While not yet ready to run quantum algorithms
with thousands of logical qubits and millions of operations, researchers have increased focus on
detailed resource estimates of potential algorithms—they are no longer content to sweep constant
factors under the big-O rug. These detailed compilations are worked out manually in a tedious and
error-prone manner.

This is the documentation for Qᴜᴀʟᴛʀᴀɴ (quantum algorithms translator):
a set of abstractions for representing quantum programs
and a library of quantum algorithms expressed in that language.



:ref:`bloq_infra`
-----------------------------

``qualtran.bloq_infra`` contains the abstractions and infrastructure for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types (``Register``), and algorithms (``CompositeBloq``). :ref:`Read more... <bloq_infra>`

.. toctree::
   :maxdepth: 2

   quantum_graph/index.rst



:ref:`bloq_algos`
------------------------------

``qualtran.bloq_algos`` contains implementations of primitive operations, quantum subroutines,
and high-level quantum programs. :ref:`Read more... <bloq_algos>`

.. toctree::
   :maxdepth: 2

   bloq_algos/index.rst


:ref:`reference`
-------------------------------

This section of the docs includes an API reference for all symbols in the library.
:ref:`Go to Reference... <reference>`

.. toctree::
   :hidden:

   reference/index.rst
