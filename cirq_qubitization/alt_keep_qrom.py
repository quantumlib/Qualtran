"""
Uses OpenFermion LCU_utils routines to generate QROM oracles for the alt and keep parts of
the SUBPREPARE oracle.
"""
from typing import List

from openfermion.circuits.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling
from cirq_qubitization.qrom import QROM


def construct_alt_keep_qrom(lcu_coefficients: List[float], probability_epsilon: float) -> QROM:
    """
    Construct the QROM that outputs alt and keep values in the SUBPREPARE routine

    Args:
        :lcu_coefficients: List of coefficients for the LCU expansion of a Hamiltonian
        :probability_epsilon: The epsilon that we use to set the precision of of the
                              subprepare approximation. This parameter is called mu.
    Returns:
        A QROM instance
    """
    alternates, keep_numers, mu = preprocess_lcu_coefficients_for_reversible_sampling(
        lcu_coefficients=lcu_coefficients, epsilon=probability_epsilon
    )
    return QROM(alternates, keep_numers)
