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

import abc
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran._infra.bloq import Bloq, DecomposeNotImplementedError
from qualtran._infra.composite_bloq import CompositeBloq
from qualtran._infra.quantum_graph import Soquet
from qualtran._infra.registers import Register, Side

if TYPE_CHECKING:
    from qualtran.cirq_interop import CirqQuregT
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.drawing import WireSymbol


def total_bits(registers: Iterable[Register]) -> int:
    """Sum of `reg.total_bits()` for each register `reg` in input `signature`."""
    return sum(reg.total_bits() for reg in registers)


def split_qubits(
    registers: Iterable[Register], qubits: Sequence[cirq.Qid]
) -> Dict[str, NDArray[cirq.Qid]]:  # type: ignore[type-var]
    """Splits the flat list of qubits into a dictionary of appropriately shaped qubit arrays."""

    qubit_regs = {}
    base = 0
    for reg in registers:
        qubit_regs[reg.name] = np.array(qubits[base : base + reg.total_bits()]).reshape(
            reg.shape + (reg.bitsize,)
        )
        base += reg.total_bits()
    return qubit_regs


def merge_qubits(
    registers: Iterable[Register],
    **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]],
) -> List[cirq.Qid]:
    """Merges the dictionary of appropriately shaped qubit arrays into a flat list of qubits."""

    ret: List[cirq.Qid] = []
    for reg in registers:
        if reg.name not in qubit_regs:
            raise ValueError(f"All qubit registers must be present. {reg.name} not in qubit_regs")
        qubits = qubit_regs[reg.name]
        qubits = np.array([qubits] if isinstance(qubits, cirq.Qid) else qubits)
        full_shape = reg.shape + (reg.bitsize,)
        if qubits.shape != full_shape:
            raise ValueError(
                f'{reg.name} register must of shape {full_shape} but is of shape {qubits.shape}'
            )
        ret += qubits.flatten().tolist()
    return ret


def get_named_qubits(registers: Iterable[Register]) -> Dict[str, NDArray[cirq.Qid]]:
    """Returns a dictionary of appropriately shaped named qubit signature for input `signature`."""

    def _qubit_array(reg: Register):
        qubits = np.empty(reg.shape + (reg.bitsize,), dtype=object)
        for ii in reg.all_idxs():
            for j in range(reg.bitsize):
                prefix = "" if not ii else f'[{", ".join(str(i) for i in ii)}]'
                suffix = "" if reg.bitsize == 1 else f"[{j}]"
                qubits[ii + (j,)] = cirq.NamedQubit(reg.name + prefix + suffix)
        return qubits

    def _qubits_for_reg(reg: Register):
        if len(reg.shape) > 0:
            return _qubit_array(reg)

        return np.array(
            [cirq.NamedQubit(f"{reg.name}")]
            if reg.total_bits() == 1
            else cirq.NamedQubit.range(reg.total_bits(), prefix=reg.name),
            dtype=object,
        )

    return {reg.name: _qubits_for_reg(reg) for reg in registers}


def _get_all_and_output_quregs_from_input(
    registers: Iterable[Register],
    qubit_manager: cirq.QubitManager,
    in_quregs: Dict[str, 'CirqQuregT'],
) -> Tuple[Dict[str, 'CirqQuregT'], Dict[str, 'CirqQuregT']]:
    """Takes care of necessary (de-/)allocations to obtain output & all qubit registers from input.

    For every register `reg` in `registers`, this method checks:
    - If `reg.side == Side.LEFT`:
        - Ensure that `in_quregs` has an entry corresponding to `reg`
        - Deallocate the corresponding qubits using `qubit_manager.deallocate`.
        - These qubits are part of `all_quregs` but not `out_quregs`.
    - If `reg.side == Side.RIGHT`:
        - Ensure that `in_quregs` does not have an entry corresponding to `reg`.
        - Allocate new multi-dimensional qubit array of shape `(*reg.shape, reg.bitsize)`.
        - These qubits are part of `all_quregs` and `out_quregs`.
    - If `reg.side == Side.THRU`:
        - Ensure that `in_quregs` has a an entry corresponding to `reg`
        - These qubits are part of `all_quregs` and `out_quregs`.

    Args:
        registers: An iterable of `Register` objects specifying the signature of a Bloq.
        qubit_manager: An instance of `cirq.QubitManager` to allocate/deallocate qubits.
        in_quregs: A dictionary mapping LEFT register names from `registers` to corresponding
            cirq-style multidimensional qubit array of shape `(*left_reg.shape, left_reg.bitsize)`.

    Returns:
        A tuple of `(all_quregs, out_quregs)`
    """
    all_quregs: Dict[str, 'CirqQuregT'] = {}
    out_quregs: Dict[str, 'CirqQuregT'] = {}
    for reg in registers:
        full_shape = reg.shape + (reg.bitsize,)
        if reg.side & Side.LEFT:
            if reg.name not in in_quregs or in_quregs[reg.name].shape != full_shape:
                # Left registers should exist as input to `as_cirq_op`.
                raise ValueError(f'Compatible {reg=} must exist in {in_quregs=}')
            all_quregs[reg.name] = in_quregs[reg.name]
        if reg.side == Side.RIGHT:
            # Right only registers will get allocated as part of `as_cirq_op`.
            if reg.name in in_quregs:
                raise ValueError(f"RIGHT register {reg=} shouldn't exist in {in_quregs=}.")
            all_quregs[reg.name] = np.array(qubit_manager.qalloc(reg.total_bits())).reshape(
                full_shape
            )
        if reg.side == Side.LEFT:
            # LEFT only registers should be de-allocated and not be part of output.
            qubit_manager.qfree(in_quregs[reg.name].flatten())

        if reg.side & Side.RIGHT:
            # Right registers should be part of the output.
            out_quregs[reg.name] = all_quregs[reg.name]
    return all_quregs, out_quregs


class GateWithRegisters(Bloq, cirq.Gate, metaclass=abc.ABCMeta):
    """`cirq.Gate`s extension with support for composite gates acting on multiple qubit registers.

    Though Cirq was nominally designed for circuit construction for near-term devices the core
    concept of the `cirq.Gate`, a programmatic representation of an operation on a state without
    a complete qubit address specification, can be leveraged to describe more abstract algorithmic
    primitives. To define composite gates, users derive from `cirq.Gate` and implement the
    `_decompose_` method that yields the sub-operations provided a flat list of qubits.

    This API quickly becomes inconvenient when defining operations that act on multiple qubit
    registers of variable sizes. Qualtran extends the `cirq.Gate` idea by introducing a new abstract
    base class `GateWithRegisters` containing abstract methods `registers` and optional
    method `decompose_from_registers` that provides an overlay to the Cirq flat address API.

    As an example, in the following code snippet we use the `GateWithRegisters` to
    construct a multi-target controlled swap operation:

    >>> import attr
    >>> import cirq
    >>> import qualtran
    >>>
    >>> @attr.frozen
    ... class MultiTargetCSwap(qualtran.GateWithRegisters):
    ...     bitsize: int
    ...
    ...     @property
    ...     def signature(self) -> qualtran.Signature:
    ...         return qualtran.Signature.build(ctrl=1, x=self.bitsize, y=self.bitsize)
    ...
    ...     def decompose_from_registers(self, context, ctrl, x, y) -> cirq.OP_TREE:
    ...         yield [cirq.CSWAP(*ctrl, qx, qy) for qx, qy in zip(x, y)]
    ...
    >>> op = MultiTargetCSwap(2).on_registers(
    ...     ctrl=[cirq.q('ctrl')],
    ...     x=cirq.NamedQubit.range(2, prefix='x'),
    ...     y=cirq.NamedQubit.range(2, prefix='y'),
    ... )
    >>> print(cirq.Circuit(op))
    ctrl: ───MultiTargetCSwap───
             │
    x0: ─────x──────────────────
             │
    x1: ─────x──────────────────
             │
    y0: ─────y──────────────────
             │
    y1: ─────y──────────────────"""

    # Part-1: Bloq interface is automatically available for users, via default convertors.

    def decompose_bloq(self) -> 'CompositeBloq':
        """Decompose this Bloq into its constituent parts contained in a CompositeBloq.

        Bloq users can call this function to delve into the definition of a Bloq. The function
        returns the decomposition of this Bloq represented as an explicit compute graph wrapped
        in a `CompositeBloq` object.

        Bloq authors can specify the bloq's decomposition by overriding any of the following two
        methods:

        - `build_composite_bloq`: Override this method to define a bloq-style decomposition using a
            `BloqBuilder` builder class to construct the `CompositeBloq` directly.
        - `decompose_from_registers`: Override this method to define a cirq-style decomposition by
            yielding cirq style operations applied on qubits.

        Irrespective of the bloq author's choice of backend to implement the
        decomposition, bloq users will be able to access both the bloq-style and Cirq-style
        interfaces. For example, users can call:

        - `cirq.decompose_once(bloq.on_registers(**cirq_quregs))`: This will yield a `cirq.OPTREE`.
            Bloqs will be wrapped in `BloqAsCirqGate` as needed.
        - `bloq.decompose_bloq()`: This will return a `CompositeBloq`.
           Cirq gates will be be wrapped in `CirqGateAsBloq` as needed.

        Thus, `GateWithRegisters` class provides a convenient way of defining objects that can be used
        interchangeably with both `Cirq` and `Bloq` constructs.

        Returns:
            A `CompositeBloq` containing the decomposition of this Bloq.

        Raises:
            DecomposeNotImplementedError: If there is no decomposition defined; namely if both:
                - `build_composite_bloq` raises a `DecomposeNotImplementedError` and
                - `decompose_from_registers` raises a `DecomposeNotImplementedError`.
        """
        from qualtran.cirq_interop._cirq_to_bloq import decompose_from_cirq_style_method

        try:
            return Bloq.decompose_bloq(self)
        except DecomposeNotImplementedError:
            return decompose_from_cirq_style_method(self)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **in_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        """Allocates/Deallocates qubits for RIGHT/LEFT only registers to construct a Cirq operation

        Args:
            qubit_manager: For allocating/deallocating qubits for RIGHT/LEFT only registers.
            in_quregs: Mapping from LEFT register names to corresponding cirq qubits.

        Returns:
            A cirq operation constructed using `self` and a mapping from RIGHT register names to
            corresponding Cirq qubits.
        """
        all_quregs, out_quregs = _get_all_and_output_quregs_from_input(
            self.signature, qubit_manager, in_quregs
        )
        return self.on_registers(**all_quregs), out_quregs

    def t_complexity(self) -> 'TComplexity':
        from qualtran.cirq_interop.t_complexity_protocol import t_complexity

        return t_complexity(self)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.cirq_interop._cirq_to_bloq import _wire_symbol_from_gate

        return _wire_symbol_from_gate(self, self.signature, soq)

    # Part-2: Cirq-FT style interface can be used to implemented algorithms by Bloq authors.

    def _num_qubits_(self) -> int:
        return total_bits(self.signature)

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        raise DecomposeNotImplementedError(f"{self} does not declare a decomposition.")

    def _decompose_with_context_(
        self, qubits: Sequence[cirq.Qid], context: Optional[cirq.DecompositionContext] = None
    ) -> cirq.OP_TREE:
        from qualtran.cirq_interop._bloq_to_cirq import _cirq_style_decompose_from_decompose_bloq

        quregs = split_qubits(self.signature, qubits)
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        try:
            return self.decompose_from_registers(context=context, **quregs)
        except DecomposeNotImplementedError:
            pass
        try:
            return _cirq_style_decompose_from_decompose_bloq(
                bloq=self, quregs=quregs, context=context
            )
        except DecomposeNotImplementedError:
            pass
        return NotImplemented

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return self._decompose_with_context_(qubits)

    def on(self, *qubits) -> 'cirq.Operation':
        import cirq

        # Multiple inheritance: use `cirq.Gate.on()`, not the bloq method.
        return cirq.Gate.on(self, *qubits)

    def on_registers(
        self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]]
    ) -> cirq.Operation:
        return self.on(*merge_qubits(self.signature, **qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Default diagram info that uses register names to name the boxes in multi-qubit gates.

        Descendants can override this method with more meaningful circuit diagram information.
        """
        wire_symbols = []
        for reg in self.signature:
            wire_symbols += [reg.name] * reg.total_bits()

        wire_symbols[0] = self.__class__.__name__
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
