import abc
from functools import cached_property
from typing import Sequence, Tuple, Iterator, List, Dict

import cirq

from cirq_qubitization import and_gate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers, Register


def _unary_iteration_segtree(
    ops: List[cirq.Operation],
    control: cirq.Qid,
    selection: Sequence[cirq.Qid],
    ancilla: Sequence[cirq.Qid],
    sl: int,
    l: int,
    r: int,
    L: int,
    R: int,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    if l >= R or L >= r:
        # Completely out of range.
        return
    if l == (r - 1):
        # Reached a leaf node; yield the operations.
        yield tuple(ops), control, l
        ops.clear()
        return
    assert sl < len(selection)
    m = (l + r) >> 1
    if R <= m:
        # Yield only left sub-tree.
        yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl + 1, l, m, L, R)
        return
    if L >= m:
        # Yield only right sub-tree
        yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl + 1, m, r, L, R)
        return
    anc, sq = ancilla[sl], selection[sl]
    ops.append(and_gate.And((1, 0)).on(control, sq, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, ancilla, sl + 1, l, m, L, R)
    ops.append(cirq.CNOT(control, anc))
    yield from _unary_iteration_segtree(ops, anc, selection, ancilla, sl + 1, m, r, L, R)
    ops.append(and_gate.And(adjoint=True).on(control, sq, anc))


def _unary_iteration_zero_controls(
    ops: List[cirq.Operation],
    selection: Sequence[cirq.Qid],
    ancilla: Sequence[cirq.Qid],
    L: int,
    R: int,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = 0, 0, 2 ** len(selection)
    m = (l + r) >> 1
    ops.extend([cirq.X(selection[0]), cirq.CNOT(selection[0], ancilla[0]), cirq.X(selection[0])])
    yield from _unary_iteration_segtree(ops, ancilla[0], selection, ancilla, sl + 1, l, m, L, R)
    ops.append(cirq.X(ancilla[0]))
    yield from _unary_iteration_segtree(ops, ancilla[0], selection, ancilla, sl + 1, m, r, L, R)
    ops.append(cirq.CNOT(selection[0], ancilla[0]))


def _unary_iteration_single_controls(
    ops: List[cirq.Operation],
    control: cirq.Qid,
    selection: Sequence[cirq.Qid],
    ancilla: Sequence[cirq.Qid],
    L: int,
    R: int,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    sl, l, r = 0, 0, 2 ** len(selection)
    yield from _unary_iteration_segtree(ops, control, selection, ancilla, sl, l, r, L, R)


def _unary_iteration_multi_controls(
    ops: List[cirq.Operation],
    controls: Sequence[cirq.Qid],
    selection: Sequence[cirq.Qid],
    ancilla: Sequence[cirq.Qid],
    L: int,
    R: int,
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    num_controls = len(controls)
    and_ancilla, and_target = ancilla[: num_controls - 2], ancilla[num_controls - 2]
    multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(
        control=controls, ancilla=and_ancilla, target=and_target
    )
    ops.append(multi_controlled_and)
    yield from _unary_iteration_single_controls(
        ops, and_target, selection, ancilla[num_controls - 1 :], L, R
    )
    ops.append(multi_controlled_and**-1)


def unary_iteration(
    L: int,
    R: int,
    ops: List[cirq.Operation],
    controls: Sequence[cirq.Qid],
    selection: Sequence[cirq.Qid],
    ancilla: Sequence[cirq.Qid],
) -> Iterator[Tuple[cirq.OP_TREE, cirq.Qid, int]]:
    """The method performs unary iteration on `selection` integer in `range(L, R)`.

    Unary iteration is a coherent for loop that can be used to conditionally perform a different
    operation on a target register for every integer in the `range(L, R)` stored in the selection
    register.

    Users can write multi-dimensional coherent for loops as follows:

    >>> N, M = 5, 7
    >>> target = [[cirq.q(f't({i}, {j})') for j in range(M)] for i in range(N)]
    >>> selection = [[cirq.q(f's({i}, {j})') for j in range(3)] for i in range(3)]
    >>> ancilla = [[cirq.q(f'a({i}, {j})') for j in range(3)] for i in range(3)]
    >>> circuit = cirq.Circuit()
    >>> i_ops = []
    >>> for i_optree, i_control, i in unary_iteration(0, N, i_ops, [], selection[0], ancilla[0]):
    >>>     circuit.append(i_optree)
    >>>     j_ops = []
    >>>     for j_optree, j_control, j in unary_iteration(0, M, j_ops, [i_control], selection[1], ancilla[1]):
    >>>         circuit.append(j_optree)
    >>>         # Conditionally perform operations on target register using `j_control`, `i` and `j`.
    >>>         circuit.append(cirq.CNOT(j_control, target[i][j]))
    >>>     circuit.append(j_ops)
    >>> circuit.append(i_ops)

    Args:
        L: Starting index of the iteration range.
        R: Ending index of the iteration range.
        ops: A list of that stores operations to be inserted in the circuit before/after
            the first/last iteration of the unary iteration for loop.
        controls: Control register of qubits.
        selection: Selection register of qubits.
        ancilla: Ancillas to be used for unary iteration and multi-controlled AND gate.

    Yields:
        (R - L) different tuples, each corresponding to a different integer in range [L, R).
        Each returned tuple also corresponds to a unique leaf in the unary iteration tree.
        The values of yielded `Tuple[cirq.OP_TREE, cirq.Qid, int]` correspond to:
        - cirq.OP_TREE: The op-tree to be inserted in the circuit to get to the current leaf.
        - cirq.Qid: Control qubit used to conditionally apply operations on the target conditioned
            on the returned integer.
        - int: The current integer in the iteration `range(L, R)`.
    """
    assert len(selection) + max(0, len(controls) - 1) == len(
        ancilla
    ), f'selection: {selection}\ncontrols: {controls}\nancilla: {ancilla}'
    assert 2 ** len(selection) >= R - L
    assert len(selection) > 0
    if len(controls) == 0:
        yield from _unary_iteration_zero_controls(ops, selection, ancilla, L, R)
    elif len(controls) == 1:
        yield from _unary_iteration_single_controls(ops, controls[0], selection, ancilla, L, R)
    elif len(controls) == 2:
        yield from _unary_iteration_multi_controls(ops, controls, selection, ancilla, L, R)


class UnaryIterationGate(GateWithRegisters):
    @cached_property
    @abc.abstractmethod
    def control_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def selection_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def target_registers(self) -> Registers:
        pass

    @cached_property
    @abc.abstractmethod
    def iteration_lengths(self) -> Tuple[int, ...]:
        pass

    @cached_property
    def ancilla_registers(self) -> Registers:
        ancillas = [
            Register(name=f'{s.name}_ancilla', bitsize=s.bitsize) for s in self.selection_registers
        ]
        ancillas[0] = Register(
            name=ancillas[0].name,
            bitsize=ancillas[0].bitsize + max(0, self.control_registers.bitsize - 1),
        )
        return Registers(ancillas)

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                *self.control_registers,
                *self.selection_registers,
                *Registers.build(ancilla=self.ancilla_registers.bitsize),
                *self.target_registers,
                *self.extra_registers,
            ]
        )

    @cached_property
    def extra_registers(self) -> Registers:
        return Registers([])

    @abc.abstractmethod
    def nth_operation(self, **kwargs) -> cirq.OP_TREE:
        """Apply nth operation on the target registers when selection registers store `n`.

        The `UnaryIterationGate` class is a mixin that represents a coherent for-loop over
        different indices (i.e. selection registers). This method denotes the "body" of the
        for-loop, which is executed `np.prod(self.iteration_lengths)` times and each iteration
        represents a unique combination of values stored in selection registers. For each call,
        the method should return the operations that should be applied to the target registers,
        given the values stored in selection registers.

        The derived classes should specify the following arguments as `**kwargs`:
            1) `control: cirq.Qid`: A qubit which can be used as a control to selectively
            apply operations when selection register stores specific value.
            2) Register names in `self.selection_registers`: Each argument corresponds to
            a selection register and represents the integer value stored in the register.
            3) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            4) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """

    def decompose_zero_selection(self, **kwargs) -> cirq.OP_TREE:
        """Specify decomposition of the gate when selection register is empty

        By default, if the selection register is empty, the decomposition will raise a
        `NotImplementedError`. The derived classes can override this method and specify
        a custom decomposition that should be used if the selection register is empty,
        i.e. `self.selection_registers.bitsize == 0`.

        The derived classes should specify the following arguments as `**kwargs`:
            1) Register names in `self.control_registers`: Each argument corresponds to a
            control register and represents sequence of qubits that represent the control register.
            2) Register names in `self.target_registers`: Each argument corresponds to a target
            register and represents the sequence of qubits that represent the target register.
            3) Register names in `self.extra_regs`: Each argument corresponds to an extra
            register and represents the sequence of qubits that represent the extra register.
        """
        raise NotImplementedError("Selection register must not be empty.")

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        if self.selection_registers.bitsize == 0:
            yield from self.decompose_zero_selection(**qubit_regs)
            return

        num_loops = len(self.iteration_lengths)
        target_regs = {k: v for k, v in qubit_regs.items() if k in self.target_registers}
        extra_regs = {k: v for k, v in qubit_regs.items() if k in self.extra_registers}
        ancilla_regs = self.ancilla_registers.split_qubits(qubit_regs['ancilla'])

        def unary_iteration_loops(
            index: int, indices: Dict[str, int], controls: Sequence[cirq.Qid]
        ) -> cirq.OP_TREE:
            if index == num_loops:
                yield self.nth_operation(
                    control=controls[0], **indices, **target_regs, **extra_regs
                )
                return
            # Use recursion to write `num_loops` nested loops using unary_iteration().
            ops = []
            ith_for_loop = unary_iteration(
                L=0,
                R=self.iteration_lengths[index],
                ops=ops,
                controls=controls,
                selection=qubit_regs[self.selection_registers[index].name],
                ancilla=ancilla_regs[self.ancilla_registers[index].name],
            )
            for op_tree, control_qid, n in ith_for_loop:
                yield op_tree
                indices[self.selection_registers[index].name] = n
                yield from unary_iteration_loops(index + 1, indices, (control_qid,))
            yield ops

        yield from unary_iteration_loops(0, {}, self.control_registers.merge_qubits(**qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Basic circuit diagram.

        Descendants are encouraged to override this with more descriptive
        circuit diagram information.
        """
        wire_symbols = ["@"] * self.control_registers.bitsize
        wire_symbols += ["In"] * self.selection_registers.bitsize
        wire_symbols += ["Anc"] * self.ancilla_registers.bitsize
        wire_symbols += [self.__class__.__name__] * self.target_registers.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
