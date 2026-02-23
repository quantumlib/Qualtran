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

**Note:** Qualtran is in beta. Some algorithms or library functionality may be incomplete
or contain inaccuracies. Open issues or contact the authors with bug reports or feedback,
and subscribe to `qualtran-announce@googlegroups.com <https://groups.google.com/g/qualtran-announce>`_
to receive the latest news and updates!


:ref:`Fundamentals <bloq_infra>`
--------------------------------

Qualtran has tools and abstractions for expressing
and reasoning about quantum algorithms, programs, and subroutines.
Our hosted language consists of Python objects representing operations (``Bloq``), quantum data
types, and algorithms.
We provide protocols for simulating algorithms, estimating resource requirements, drawing diagrams,
and more.

The :ref:`Fundamentals<bloq_infra>` section provides documentation for these tools and abstractions.

.. toctree::
   :maxdepth: 2
   :hidden:

   bloq_infra.rst



:ref:`Bloqs Library <bloqs_library>`
------------------------------------

Qualtran also contains a library of quantum subroutines as well as high-level quantum programs.
These primitives can be imported from the ``qualtran.bloqs`` subpackage and are documented
in the :ref:`Bloqs Library<bloqs_library>` docs section.

.. toctree::
   :maxdepth: 2
   :hidden:

   bloqs/index.rst


:ref:`API Reference <reference>`
--------------------------------

Each class, method, and function in the library is documented in the
:ref:`API Reference<reference>`.

.. toctree::
   :hidden:

   reference/index.rst


Installation
------------

Basics
~~~~~~

Qualtran is a Python package. If you already have a working Python environment,

.. code-block:: sh

    pip install qualtran

will install the library and its dependencies. Qualtran uses `Graphviz <https://graphviz.org/>`_
to draw diagrams, which can't be installed using the Python package manager ``pip``. Follow
the `Graphviz installation instructions <https://graphviz.org/download/>`_ to set it up.
Qualtran requires Python 3.11 or newer.

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

Source code is hosted on `GitHub <https://github.com/quantumlib/Qualtran>`_. You may wish
to inspect the source code for the bloqs provided in the library or interactively execute
the Jupyter notebooks included in the repository. In this case, you can clone the git repository
and install from source.

.. code-block:: sh

    git clone https://github.com/quantumlib/Qualtran.git
    cd Qualtran/
    pip install -e .

If you plan on developing Qualtran or you would just like to install all optional and developer
dependencies, you can run

.. code-block:: sh

    pip install -r dev_tools/requirements/envs/dev.env.txt

from the repository root.


Setting up a Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many ways to install Python and Qualtran's dependencies. This section will provide
a walkthrough of one particular way.

First, download and install `Miniconda <https://docs.anaconda.com/free/miniconda/>`_. This is
a small tool that lets you install and manage Python environments. It supports different
versions of Python in each environment, and can be installed without Administrator privileges.
After following the installation instructions for Miniconda on your platform, you can create
and activate an environment for Qualtran:

.. code-block:: sh

    conda create -n my-qualtran-env python=3.11
    conda activate my-qualtran-env

That code snippet will create a new environment with the given name (choose whatever you like)
and install Python into it. Each time you start a new terminal you must first activate
your environment

.. code-block:: sh

    conda activate my-qualtran-env

You can use conda directly to install Graphviz:

.. code-block:: sh

    conda install graphviz

At this point, you have an isolated, modern Python installation with Graphviz installed. From here,
you can follow the basic installation instructions.

.. code-block:: sh

    pip install qualtran

Next Steps
~~~~~~~~~~

You should be able to import ``qualtran`` from a Python command prompt.

.. code-block:: python

    import qualtran

If this is successful,
you can move on to learning how to `write bloqs <./_infra/Bloqs-Tutorial.html>`_ or investigate
the :ref:`bloqs library<bloqs_library>`.
