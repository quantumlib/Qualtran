from typing import Sequence, List, Tuple, Dict

import cirq
import numpy as np
import quimb
import quimb.tensor as qtn


def _get_basis_state_data(x: int, q: cirq.Qid):
    """Helper function to return the state vector representation for a computational basis state.

    Args:
        x: 0 or 1
        q: The qubit (used for error message only).
    """
    if x == 0:
        return quimb.up().squeeze()
    elif x == 1:
        return quimb.down().squeeze()
    else:
        raise ValueError(f"Unknown state {x} for {q}")


def circuit_to_tensors(
    circuit: cirq.AbstractCircuit,
    initial_state: Dict[cirq.Qid, int],
    final_state: Dict[cirq.Qid, int],
) -> Tuple[List[qtn.Tensor], Dict['cirq.Qid', int], Dict[str, Tuple[float, float]]]:
    """Given a circuit, construct a tensor network representation.

    We use a convention for the names of the tensors' indices: 'i{index}_{q}' where
    `index` is a numeric index along the qubit frontier and `q` is the qubit.

    We tag tensors with:
        - The bra or ket initialization value, where applicable
        - The number of qubits in a unitary: 'U{len(qubits)}'
        - a unique tag for the tensor used for positioning information equal to
          '{mi}_{qubits[0]}', namely: the moment index and a representative qubit from the
          operation. Initialization kets are at mi=-1

    Args:
        circuit: The circuit containing operations that implement the
            cirq.unitary() protocol.
        initial_state: A mapping from qubit to 0 or 1 setting the initial kets. If a qubit
            is in the circuit but not in intial_state, the index will be left open.
        final_state: A mapping from qubit to 0 or 1 setting the final bras. If a qubit
            is in the circuit but not in final_state, the index will be left open.

    Returns:
        tensors: A list of quimb Tensor objects
        qubit_frontier: A mapping from qubit to time index at the end of
            the circuit. This can be used to deduce the names of the free
            tensor indices.
        positions: A mapping suitable for `tn.draw(fix=positions)` to lay out tensors
            in a diagram. This is a mapping from positioning tag names (see above) to (x, y)
            coordinates.
    """
    all_qubits = sorted(circuit.all_qubits() | set(initial_state.keys()))
    qubit_frontier = {q: 0 for q in all_qubits}
    tensors: List[qtn.Tensor] = []

    positions: Dict[str, Tuple[float, float]] = {}
    x_scale = 2
    y_scale = 3

    def _add_tensor(
        data: np.ndarray, qubits: Sequence[cirq.Qid], mi: int, thru_tensor=True, tag=None
    ):
        """Add a tensor to `tensors` and include its position in `positions`.

        This function introduces the convention for index name: 'i{index}_{q}' where
        `index` is a numeric index along the qubit frontier and `q` is the qubit.

        This function tags tensors with:
            - a caller-specified tag.
            - a unique tag for the tensor used for positioning information equal to
              '{mi}_{qubits[0]}', namely: the moment index and a representative qubit from the
              operation. Initialization kets are at mi=-1

        Args:
            data: The tensor data
            qubits: Qubits participating in this tensor
            mi: The moment index
            thru_tensor: If True (the default), we have input and output legs on the tensor.
                Otherwise, we only have one. Gates are thru-tensors whereas bras and kets
                are one-sided.
            tag: A custom tag for this tensor.
        """
        start_inds = [f'i{qubit_frontier[q]}_{q}' for q in qubits]

        if thru_tensor:
            # We have input and output legs: advance the frontier and include
            # the second set of indices.
            for q in qubits:
                qubit_frontier[q] += 1
            end_inds = [f'i{qubit_frontier[q]}_{q}' for q in qubits]
        else:
            # Only one set of indices, which is `start_inds`.
            end_inds = []

        tid_tag = f'{mi}_{qubits[0]}'

        qinds = [all_qubits.index(q) for q in qubits]
        qy = np.mean(qinds).item()
        qy = len(all_qubits) - qy - 1

        positions[tid_tag] = (mi * x_scale, qy * y_scale)
        t = qtn.Tensor(data=data, inds=end_inds + start_inds, tags={tag, tid_tag})
        tensors.append(t)

    for q, init in initial_state.items():
        _add_tensor(
            data=_get_basis_state_data(init, q), qubits=[q], mi=-1, thru_tensor=False, tag=f'{init}'
        )

    for mi, moment in enumerate(circuit.moments):
        for op in moment.operations:
            assert cirq.has_unitary(op.gate)
            U = cirq.unitary(op).reshape((2,) * 2 * len(op.qubits))
            _add_tensor(data=U, qubits=op.qubits, mi=mi, tag=f'U{len(op.qubits)}')

    for q, final in final_state.items():
        _add_tensor(
            data=_get_basis_state_data(final, q),
            qubits=[q],
            mi=len(circuit),
            thru_tensor=False,
            tag=f'{final}',
        )

    return tensors, qubit_frontier, positions
