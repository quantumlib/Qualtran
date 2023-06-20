# !!!! Do not modify imports !!!!
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from typing import *

import cirq
import cirq_ft
import cirq_ft.infra.testing as cq_testing
import numpy as np

import cirq_qubitization

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


def _make_ApplyGateToLthQubit():
    from cirq_ft import ApplyGateToLthQubit, Registers, SelectionRegisters

    def _z_to_odd(n: int):
        if n % 2 == 1:
            return cirq.Z
        return cirq.I

    apply_z_to_odd = ApplyGateToLthQubit(
        SelectionRegisters.build(selection=(3, 4)),
        nth_gate=_z_to_odd,
        control_regs=Registers.build(control=2),
    )

    return apply_z_to_odd


def _make_QROM():
    from cirq_ft import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=(3,), target_bitsizes=(3,))


def _make_MultiTargetCSwap():
    from cirq_ft import MultiTargetCSwap

    return MultiTargetCSwap(3)


def _make_MultiTargetCSwapApprox():
    from cirq_ft import MultiTargetCSwapApprox

    return MultiTargetCSwapApprox(2)


def _make_SwapWithZeroGate():
    from cirq_ft import SwapWithZeroGate

    return SwapWithZeroGate(selection_bitsize=2, target_bitsize=3, n_target_registers=4)


def _make_GenericSelect():
    from cirq_ft import GenericSelect

    target_bitsize = 4
    us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
    us = [cirq.DensePauliString(u) for u in us]
    selection_bitsize = int(np.ceil(np.log2(len(us))))
    return GenericSelect(selection_bitsize, target_bitsize, select_unitaries=us)


def _make_StatePreparationAliasSampling():
    from cirq_ft import StatePreparationAliasSampling

    coeffs = np.array([1.0, 1, 3, 2])
    mu = 3

    state_prep = StatePreparationAliasSampling.from_lcu_probs(
        coeffs, probability_epsilon=2**-mu / len(coeffs)
    )
    return state_prep


def _make_QubitizationWalkOperator():
    from cirq_ft.algos.qubitization_walk_operator_test import get_walk_operator_for_1d_ising_model

    return get_walk_operator_for_1d_ising_model(4, 2e-1)
