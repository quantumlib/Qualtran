from typing import Sequence

import cirq
import numpy as np


def assert_circuit_inp_out_cirqsim(
    circuit: cirq.AbstractCircuit,
    qubits: Sequence[cirq.Qid],
    inp: Sequence[int],
    out: Sequence[int],
    decimals: int = 2,
):
    """Use a Cirq simulator to test that `circuit` behaves correctly on an input.

    Args:
        circuit: The circuit representing the reversible classical operation.
        qubits: The qubits in a definite order.
        inp: The input state bit values.
        out: The (correct) output state bit values.
        decimals: The number of decimals of precision to use when comparing
            amplitudes. Reversible classical operations should produce amplitudes
            that are 0 or 1.
    """
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=inp, qubit_order=qubits
    )
    actual = result.dirac_notation(decimals=decimals)[1:-1]
    should_be = "".join(str(x) for x in out)
    assert actual == should_be, (actual, should_be)
