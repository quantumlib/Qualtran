# !!!! Do not modify imports !!!!
# pylint: disable=unused-import,wildcard-import,unused-wildcard-import
from typing import *

import cirq
import numpy as np

import cirq_qubitization
import cirq_qubitization.cirq_infra.testing as cq_testing

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
    from cirq_qubitization.cirq_algos.apply_gate_to_lth_target import ApplyGateToLthQubit
    from cirq_qubitization.cirq_infra.gate_with_registers import Registers, SelectionRegisters

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
    from cirq_qubitization.cirq_algos import QROM

    return QROM([np.array([1, 2, 3, 4, 5])], selection_bitsizes=[3], target_bitsizes=[3])


def _make_MultiTargetCSwap():
    from cirq_qubitization.cirq_algos.swap_network import MultiTargetCSwap

    return MultiTargetCSwap(3)


def _make_MultiTargetCSwapApprox():
    from cirq_qubitization.cirq_algos.swap_network import MultiTargetCSwapApprox

    return MultiTargetCSwapApprox(2)


def _make_SwapWithZeroGate():
    from cirq_qubitization.cirq_algos.swap_network import SwapWithZeroGate

    return SwapWithZeroGate(selection_bitsize=2, target_bitsize=3, n_target_registers=4)


def _make_GenericSelect():
    from cirq_qubitization.generic_select import GenericSelect

    target_bitsize = 4
    us = ['XIXI', 'YIYI', 'ZZZZ', 'ZXYZ']
    us = [cirq.DensePauliString(u) for u in us]
    selection_bitsize = int(np.ceil(np.log2(len(us))))
    return GenericSelect(selection_bitsize, target_bitsize, select_unitaries=us)


def _make_StatePreparationAliasSampling():
    from cirq_qubitization.cirq_algos.state_preparation import StatePreparationAliasSampling

    coeffs = np.array([1.0, 1, 3, 2])
    mu = 3

    return StatePreparationAliasSampling.from_lcu_probs(
        coeffs, probability_epsilon=2**-mu / len(coeffs)
    )


def _make_QubitizationWalkOperator():
    from cirq_qubitization.cirq_algos.qubitization_walk_operator_test import (
        get_walk_operator_for_1d_ising_model,
    )

    return get_walk_operator_for_1d_ising_model(4, 2e-1)
