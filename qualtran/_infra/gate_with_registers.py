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
from qualtran._infra.registers import Register

if TYPE_CHECKING:
    from qualtran.cirq_interop import CirqQuregT


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
        from qualtran.cirq_interop._cirq_to_bloq import decompose_from_cirq_op

        try:
            return Bloq.decompose_bloq(self)
        except DecomposeNotImplementedError:
            return decompose_from_cirq_op(self, decompose_once=True)

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        from qualtran.cirq_interop._bloq_to_cirq import _construct_op_from_gate

        return _construct_op_from_gate(
            self,
            in_quregs={k: np.array(v) for k, v in cirq_quregs.items()},
            qubit_manager=qubit_manager,
        )

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
        qubit_regs = split_qubits(self.signature, qubits)
        if context is None:
            context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
        try:
            return self.decompose_from_registers(context=context, **qubit_regs)
        except DecomposeNotImplementedError as e:
            pass
        try:
            qm = context.qubit_manager
            return Bloq.decompose_bloq(self).to_cirq_circuit(qubit_manager=qm, **qubit_regs)[0]
        except DecomposeNotImplementedError as e:
            pass
        return NotImplemented

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return self._decompose_with_context_(qubits)

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
