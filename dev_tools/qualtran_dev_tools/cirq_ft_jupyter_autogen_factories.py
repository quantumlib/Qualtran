#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# !!!! Do not modify imports !!!!
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from typing import *

import cirq
import numpy as np

import qualtran
import qualtran.cirq_interop.testing as cq_testing

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


def _make_QROM():
    from qualtran.bloqs.qrom import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=(3,), target_bitsizes=(3,))


def _make_MultiTargetCSwap():
    from qualtran.bloqs.basic_gates import CSwap

    return CSwap(3)


def _make_MultiTargetCSwapApprox():
    from qualtran.bloqs.swap_network import CSwapApprox

    return CSwapApprox(2)


def _make_SwapWithZeroGate():
    from qualtran.bloqs.swap_network import SwapWithZero

    return SwapWithZero(selection_bitsize=2, target_bitsize=3, n_target_registers=4)


def _make_SelectPauliLCU():
    from qualtran.bloqs.select_pauli_lcu import SelectPauliLCU

    target_bitsize = 4
    us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
    us = [cirq.DensePauliString(u) for u in us]
    selection_bitsize = int(np.ceil(np.log2(len(us))))
    return SelectPauliLCU(selection_bitsize, target_bitsize, select_unitaries=us)


def _make_StatePreparationAliasSampling():
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling

    coeffs = np.array([1.0, 1, 3, 2])
    mu = 3

    state_prep = StatePreparationAliasSampling.from_lcu_probs(
        coeffs, probability_epsilon=2**-mu / len(coeffs)
    )
    return state_prep


def _make_QubitizationWalkOperator():
    from qualtran.bloqs.qubitization_walk_operator_test import get_walk_operator_for_1d_Ising_model

    return get_walk_operator_for_1d_Ising_model(4, 2e-1)
