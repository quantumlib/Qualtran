"""Cirq-to-Qualtran circuit conversion for FLASQ analysis."""
from typing import Dict, Tuple, Union, cast, List, Optional
import warnings
import qualtran.cirq_interop
import cirq
import numpy as np

from qualtran import Bloq, Signature, CompositeBloq
from qualtran.cirq_interop import cirq_gate_to_bloq, cirq_optree_to_cbloq
from qualtran.bloqs.mcmt import And

from qualtran.surface_code.flasq.span_counting import BloqWithSpanInfo, calculate_spans


def _get_coords_from_op(op: cirq.Operation) -> List[Tuple[int, ...]]:
    """Extracts a list of (row, col) or (x, 0) coordinates from a cirq.Operation.

    The cirq operation must act only on GridQubits or only on LineQubits.

    Raises:
        TypeError: If the operation acts on unsupported qubit types or mixes types.
        ValueError: If an operation mixes LineQubits and GridQubits.
    """
    qubits: Tuple[cirq.Qid, ...] = op.qubits
    if not qubits:
        return []

    first_qubit = qubits[0]
    if isinstance(first_qubit, cirq.GridQubit):
        if not all(isinstance(q, cirq.GridQubit) for q in qubits):
            raise ValueError(f"Operation {op} mixes qubit types.")
        return [(q.row, q.col) for q in cast(Tuple[cirq.GridQubit, ...], qubits)]
    elif isinstance(first_qubit, cirq.LineQubit):
        if not all(isinstance(q, cirq.LineQubit) for q in qubits):
            raise ValueError(f"Operation {op} mixes qubit types.")
        return [(q.x, 0) for q in cast(Tuple[cirq.LineQubit, ...], qubits)]
    else:
        raise TypeError(
            f"Operation {op} acts on unsupported qubit type: {type(first_qubit)}. "
            f"Only LineQubit and GridQubit are supported for span calculation."
        )


def cirq_op_to_bloq_tolerate_classical_controls(op: cirq.Operation) -> Bloq:
    """Converts a Cirq operation to a Bloq, tolerating classical controls.

    Specifically, we just assume that the classically controlled op will always
    be performed. This can help us get an acceptable upper bound on costs although
    it will not always be correct.
    """
    if isinstance(op, cirq.ClassicallyControlledOperation):
        return qualtran.cirq_interop._cirq_to_bloq._extract_bloq_from_op(
            op.without_classical_controls()
        )

    else:
        return qualtran.cirq_interop._cirq_to_bloq._extract_bloq_from_op(op)


def cirq_op_to_bloq_with_span(
    op: cirq.Operation, tolerate_classical_controls: bool = False
) -> Bloq:
    """Converts a Cirq operation to a Bloq, adding span info if multi-qubit.

    Single-qubit operations are returned as Bloq using cirq_gate_to_bloq.
    Multi-qubit gates are wrapped using BloqWithSpanInfo, calculating the span
    using `span_counting.calculate_spans`. If span calculation fails, a warning
    is issued and the base bloq is returned without wrapping.

    Args:
        op: The cirq.Operation to convert.
        tolerate_classical_controls: If True, classically controlled operations
            are converted by ignoring the classical control condition.

    Returns:
        A Bloq, potentially wrapped in BloqWithSpanInfo.

    Raises:
        ValueError: If the operation has no gate and `tolerate_classical_controls` is False.
    """
    if tolerate_classical_controls:
        base_bloq = cirq_op_to_bloq_tolerate_classical_controls(op)
    else:
        if op.gate is None:
            raise ValueError(f"Operation {op} has no gate, cannot convert to Bloq.")

        base_bloq = cirq_gate_to_bloq(op.gate)
    n_qubits = len(op.qubits)

    if n_qubits <= 1:
        return base_bloq
    else:
        try:
            coords = _get_coords_from_op(op)
            connect_span, compute_span = calculate_spans(coords, base_bloq)
            return BloqWithSpanInfo(
                wrapped_bloq=base_bloq,
                connect_span=connect_span,
                compute_span=compute_span,
            )
        except (TypeError, ValueError, NotImplementedError) as e:
            warnings.warn(
                f"Could not calculate span for {op}: {e}. Returning base bloq without span info."
            )
            return base_bloq


def flasq_intercepting_decomposer(op: cirq.Operation) -> List[cirq.Operation]:
    """Intercepts cirq.decompose and keeps things FLASQ-friendly"""
    gate = op.gate
    if isinstance(gate, cirq.ZZPowGate):
        a, b = op.qubits
        exponent = gate.exponent
        return [
            cirq.CNOT.on(a, b),
            cirq.ZPowGate(exponent=op.gate.exponent).on(b),
            cirq.CNOT.on(a, b),
        ]

    return NotImplemented


def flasq_decompose_keep(op: cirq.Operation) -> bool:
    """Stops cirq.decompose from decomposing FLASQ gates.

    Args:
        op: The cirq.Operation to check.

    Returns:
        True if the operation is a FLASQ primitive that we don't want to decompose.
    """
    if op.gate in [cirq.CNOT, cirq.CZ, cirq.H, cirq.T, cirq.TOFFOLI, cirq.SWAP]:
        return True

    if isinstance(op.gate, And):
        return True

    return False


def convert_circuit_for_flasq_analysis(
    circuit: cirq.Circuit,
    signature: Optional[Signature] = None,
    in_quregs: Optional[Dict[str, "CirqQuregT"]] = None,
    out_quregs: Optional[Dict[str, "CirqQuregT"]] = None,
    qubit_manager=None,
) -> Tuple[CompositeBloq, cirq.Circuit]:
    """Uses a special set of decomposition rules for FLASQ analysis."""

    # op_tree = cirq.decompose(circuit, intercepting_decomposer=flasq_intercepting_decomposer)
    if qubit_manager is not None:
        context = cirq.DecompositionContext(qubit_manager=qubit_manager)
    else:
        context = None

    op_tree = cirq.decompose(
        circuit,
        intercepting_decomposer=flasq_intercepting_decomposer,
        keep=flasq_decompose_keep,
        on_stuck_raise=None,
        context=context,
    )

    # Useful to return for analysis / testing.
    decomposed_circuit = cirq.Circuit(op_tree)

    cbloq = cirq_optree_to_cbloq(
        op_tree,
        signature=signature,
        in_quregs=in_quregs,
        out_quregs=out_quregs,
        op_conversion_method=cirq_op_to_bloq_with_span,
    )

    return cbloq, decomposed_circuit
