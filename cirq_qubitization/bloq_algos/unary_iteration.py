from functools import cached_property
from typing import *  # TODO
from typing import Any, Dict, Tuple, TYPE_CHECKING

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from cirq_qubitization import cirq_infra
from cirq_qubitization.bloq_algos.and_bloq import And
from cirq_qubitization.bloq_algos.basic_gates import CNOT, XGate
from cirq_qubitization.bloq_algos.basic_gates.swap import CSwap
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters
from cirq_qubitization.quantum_graph.quantum_graph import Soquet
from cirq_qubitization.t_complexity_protocol import t_complexity, TComplexity

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.cirq_conversion import CirqQuregT


def _unary_iteration_segtree(
    bb: CompositeBloqBuilder,
    func,
    control: Soquet,
    selection: NDArray[Soquet],
    l: int,
    r: int,
    l_iter: int,
    r_iter: int,
) -> Tuple[Soquet, NDArray[Soquet]]:
    if l >= r_iter or l_iter >= r:
        # Range corresponding to this node is completely outside of iteration range.
        raise NotImplementedError()
        return control, selection
    if l == (r - 1):
        assert len(selection) == 0, selection
        control = func(bb, l, control)
        # yield l
        return control, selection
    m = (l + r) >> 1
    if r_iter <= m:
        # Yield only left sub-tree.
        sq = selection[0]
        subsel = selection[1:]
        control, subsel = _unary_iteration_segtree(bb, func, control, subsel, l, m, l_iter, r_iter)
        return control, np.concatenate(([sq], subsel))
    if l_iter >= m:
        sq = selection[0]
        subsel = selection[1:]
        control, subsel = _unary_iteration_segtree(bb, func, control, subsel, m, r, l_iter, r_iter)
        return control, np.concatenate(([sq], subsel))

    sq = selection[0]
    subsel = selection[1:]
    (control, sq), anc = bb.add(And(1, 0), ctrl=np.array([control, sq]))
    anc, subsel = _unary_iteration_segtree(bb, func, anc, subsel, l, m, l_iter, r_iter)
    control, anc = bb.add(CNOT(), ctrl=control, target=anc)
    anc, subsel = _unary_iteration_segtree(bb, func, anc, subsel, m, r, l_iter, r_iter)
    ((control, sq),) = bb.add(And(adjoint=True), ctrl=np.array([control, sq]), target=anc)
    selection = np.concatenate(([sq], subsel))
    return control, selection


def _unary_iteration_single_control(
    bb: CompositeBloqBuilder, control: Soquet, selection: NDArray[Soquet], l_iter: int, r_iter: int
) -> Tuple[Soquet, NDArray[Soquet]]:
    l = 0
    r = 2 ** len(selection)
    control, selection = _unary_iteration_segtree(bb, control, selection, l, r, l_iter, r_iter)
    return control, selection


def _unary_iteration_multi_controls(
    ops: List[cirq.Operation],
    controls: Sequence[cirq.Qid],
    selection: Sequence[cirq.Qid],
    l_iter: int,
    r_iter: int,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    num_controls = len(controls)
    and_ancilla = cirq_infra.qalloc(num_controls - 2)
    and_target = cirq_infra.qalloc(1)[0]
    multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(
        control=controls, ancilla=and_ancilla, target=and_target
    )
    ops.append(multi_controlled_and)
    yield from _unary_iteration_single_control(ops, and_target, selection, l_iter, r_iter)
    ops.append(multi_controlled_and**-1)
    cirq_infra.qfree(and_ancilla)


def unary_iteration(
    l_iter: int, r_iter: int, controls: Sequence[cirq.Qid], selection: Sequence[cirq.Qid]
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    """The method performs unary iteration on `selection` integer in `range(l_iter, r_iter)`.

    Unary iteration is a coherent for loop that can be used to conditionally perform a different
    operation on a target register for every integer in the `range(l_iter, r_iter)` stored in the
    selection register.

    Users can write multi-dimensional coherent for loops as follows:

    >>> N, M = 5, 7
    >>> target = [[cirq.q(f't({i}, {j})') for j in range(M)] for i in range(N)]
    >>> selection = [[cirq.q(f's({i}, {j})') for j in range(3)] for i in range(3)]
    >>> circuit = cirq.Circuit()
    >>> i_ops = []
    >>> for i_optree, i_control, i in unary_iteration(0, N, i_ops, [], selection[0]):
    >>>     circuit.append(i_optree)
    >>>     j_ops = []
    >>>     for j_optree, j_control, j in unary_iteration(0, M, j_ops, [i_control], selection[1]):
    >>>         circuit.append(j_optree)
    >>>         # Conditionally perform operations on target register using `j_control`, `i` and `j`.
    >>>         circuit.append(cirq.CNOT(j_control, target[i][j]))
    >>>     circuit.append(j_ops)
    >>> circuit.append(i_ops)

    Args:
        l_iter: Starting index of the iteration range.
        r_iter: Ending index of the iteration range.
        flanking_ops: A list of `cirq.Operation`s that represents operations to be inserted in the
            circuit before/after the first/last iteration of the unary iteration for loop. Note that
            the list is mutated by the function, such that before calling the function, the list
            represents operations to be inserted before the first iteration and after the last call
            to the function, list represents operations to be inserted at the end of last iteration.
        controls: Control register of qubits.
        selection: Selection register of qubits.

    Yields:
        (r_iter - l_iter) different tuples, each corresponding to an integer in range
        [l_iter, r_iter).
        Each returned tuple also corresponds to a unique leaf in the unary iteration tree.
        The values of yielded `Tuple[cirq.OP_TREE, cirq.Qid, int]` correspond to:
        - cirq.OP_TREE: The op-tree to be inserted in the circuit to get to the current leaf.
        - cirq.Qid: Control qubit used to conditionally apply operations on the target conditioned
            on the returned integer.
        - int: The current integer in the iteration `range(l_iter, r_iter)`.
    """
    assert 2 ** len(selection) >= r_iter - l_iter
    assert len(selection) > 0
    if len(controls) == 0:
        yield from _unary_iteration_zero_control(flanking_ops, selection, l_iter, r_iter)
    elif len(controls) == 1:
        yield from _unary_iteration_single_control(
            flanking_ops, controls[0], selection, l_iter, r_iter
        )
    else:
        yield from _unary_iteration_multi_controls(
            flanking_ops, controls, selection, l_iter, r_iter
        )


class UnaryIterator:
    selection_registers: FancyRegisters
    target_registers: FancyRegisters
    iteration_lengths: Tuple[int, ...]

    def decompose_helper(self, bb, qubit_regs):
        num_loops = len(self.iteration_lengths)
        target_regs = {qubit_regs[reg.name] for reg in self.target_registers}

        # extra_regs = {k: v for k, v in qubit_regs.items() if k in self.extra_registers}

        def unary_iteration_loops(nested_depth: int):
            """Recursively write any number of nested coherent for-loops using unary iteration.

            This helper method is useful to write `num_loops` number of nested coherent for-loops by
            recursively calling this method `num_loops` times. The ith recursive call of this method
            has `nested_depth=i` and represents the body of ith nested for-loop.

            Args:
                nested_depth: Integer between `[0, num_loops]` representing the nest-level of
                    for-loop for which this method implements the body.
                selection_reg_name_to_val: A dictionary containing `nested_depth` elements mapping
                    the selection integer names (i.e. loop variables) to corresponding values;
                    for each of the `nested_depth` parent for-loops written before.
                controls: Control qubits that should be used to conditionally activate the body of
                    this for-loop.

            Returns:
                `cirq.OP_TREE` implementing `num_loops` nested coherent for-loops, with operations
                returned by `self.nth_operation` applied conditionally to the target register based
                on values of selection registers.
            """
            if nested_depth == num_loops:
                yield ctrls[0], selection_reg_name_to_val
                return

            # Use recursion to write `num_loops` nested loops using unary_iteration().
            ops = []
            ith_for_loop = unary_iteration(
                l_iter=0,
                r_iter=self.iteration_lengths[nested_depth],
                flanking_ops=ops,
                controls=controls,
                selection=qubit_regs[self.selection_registers[nested_depth].name],
            )
            for op_tree, control_qid, n in ith_for_loop:
                yield op_tree
                selection_reg_name_to_val[self.selection_registers[nested_depth].name] = n
                yield from unary_iteration_loops(
                    nested_depth + 1, selection_reg_name_to_val, (control_qid,)
                )
            yield ops

        yield from unary_iteration_loops(0, {}, self.control_registers.merge_qubits(**qubit_regs))


@frozen
class UnaryIteration(Bloq):
    control_registers: FancyRegisters
    selection_registers: FancyRegisters
    target_registers: FancyRegisters
    extra_registers: FancyRegisters
    iteration_lengths: Tuple[int, ...]

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.target_registers,
                *self.extra_registers,
            ]
        )
