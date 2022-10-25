# !!!! Do not modify imports !!!!
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from typing import *
import cirq
import numpy as np
import cirq_qubitization
import cirq_qubitization.testing as cq_testing

# pylint: enable=unused-import,wildcard-import,unused-wildcard-import
# !!!! Do not modify imports !!!!

# This module contains functions that generate demo gate objects. Both the code and
# the objects they return are used to render some auto-generated cells in our jupyter
# notebooks.
#
# The above imports are the imports guaranteed to be in a jupyter notebook. For bespoke
# imports, use a local import in your function. This will be rendered into the notebook as well.
#
# These functions must have a globally unique name, take no arguments, and finish with
# a `return` statement from which we can extract an expression that will be rendered into
# a notebook template.
