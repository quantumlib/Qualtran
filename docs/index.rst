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

**Note:** Qualtran is an experimental preview release. We provide no backwards compatibility
guarantees. Some algorithms or library functionality may be incomplete or contain inaccuracies.
Open issues or contact the authors with bug reports or feedback.

Subscribe to `qualtran-announce@googlegroups.com <https://groups.google.com/g/qualtran-announce>`_
to receive the latest news and updates!


:ref:`bloq_infra`
-----------------------------

``qualtran`` and its submodules contain the abstractions and infrastructure for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types (``Register``), and algorithms (``CompositeBloq``). :ref:`Read more... <bloq_infra>`

.. toctree::
   :maxdepth: 2
   :hidden:

   bloq_infra.rst



:ref:`bloqs_library`
------------------------------

``qualtran.bloqs`` contains implementations of primitive operations, quantum subroutines,
and high-level quantum programs. :ref:`Read more... <bloqs_library>`

.. toctree::
   :maxdepth: 2
   :hidden:

   bloqs/index.rst


:ref:`reference`
-------------------------------

This section of the docs includes an API reference for all symbols in the library.
:ref:`Go to Reference... <reference>`

.. toctree::
   :hidden:

   reference/index.rst
