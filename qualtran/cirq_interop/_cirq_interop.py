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

"""Functionality for the `Bloq.as_cirq_op(...)` protocol"""

import itertools
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cirq
import cirq_ft
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    BloqInstance,
    CompositeBloq,
    Connection,
    DanglingT,
    LeftDangle,
    Register,
    RightDangle,
    Side,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.composite_bloq import _binst_to_cxns
from qualtran.bloqs.util_bloqs import Allocate, Free

CirqQuregT = NDArray[cirq.Qid]
CirqQuregInT = Union[NDArray[cirq.Qid], Sequence[cirq.Qid]]


def signature_from_cirq_registers(registers: Iterable[cirq_ft.Register]) -> 'Signature':
    return Signature([Register(reg.name, bitsize=1, shape=reg.shape) for reg in registers])


@frozen
class CirqGateAsBloq(Bloq):
    """A Bloq wrapper around a `cirq.Gate`.

    This bloq has one thru-register named "qubits", which is a 1D array of soquets
    representing individual qubits.
    """

    gate: cirq.Gate

    def pretty_name(self) -> str:
        return f'cirq.{self.gate}'

    def short_name(self) -> str:
        g = min(self.gate.__class__.__name__, str(self.gate), key=len)
        return f'cirq.{g}'

    @cached_property
    def signature(self) -> 'Signature':
        return signature_from_cirq_registers(self.cirq_registers)

    @cached_property
    def cirq_registers(self) -> cirq_ft.Registers:
        if isinstance(self.gate, cirq_ft.GateWithRegisters):
            return self.gate.registers
        else:
            return cirq_ft.Registers.build(qubits=cirq.num_qubits(self.gate))

    def decompose_bloq(self) -> 'CompositeBloq':
        quregs = self.signature.get_cirq_quregs()
        qubit_manager = cirq.ops.SimpleQubitManager()
        cirq_op, quregs = self.as_cirq_op(qubit_manager, **quregs)
        context = cirq.DecompositionContext(qubit_manager=qubit_manager)
        decomposed_optree = cirq.decompose_once(cirq_op, context=context, default=None)
        if decomposed_optree is None:
            raise NotImplementedError(f"{self} does not support decomposition.")
        return cirq_optree_to_cbloq(decomposed_optree, signature=self.signature, cirq_quregs=quregs)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        unitary = cirq.unitary(self.gate).reshape((2,) * 2 * self.cirq_registers.total_bits())
        incoming_list = [
            *itertools.chain.from_iterable(
                [np.array(incoming[reg.name]).flatten() for reg in self.signature.lefts()]
            )
        ]
        outgoing_list = [
            *itertools.chain.from_iterable(
                [np.array(outgoing[reg.name]).flatten() for reg in self.signature.rights()]
            )
        ]

        tn.add(
            qtn.Tensor(
                data=unitary, inds=outgoing_list + incoming_list, tags=[self.short_name(), tag]
            )
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        merged_qubits = np.concatenate(
            [cirq_quregs[reg.name].flatten() for reg in self.signature.lefts()]
        )
        assert len(merged_qubits) == cirq.num_qubits(self.gate)
        return self.gate.on(*merged_qubits), cirq_quregs

    def t_complexity(self) -> 'cirq_ft.TComplexity':
        return cirq_ft.t_complexity(self.gate)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        wire_symbols = cirq.circuit_diagram_info(self.gate).wire_symbols
        begin = 0
        symbol: str = soq.pretty()
        for reg in self.signature:
            finish = begin + np.product(reg.shape)
            if reg == soq.reg:
                symbol = np.array(wire_symbols[begin:finish]).reshape(reg.shape)[soq.idx]
            begin = finish
        return directional_text_box(text=symbol, side=soq.reg.side)


def _split_qvars_for_regs(
    qvars: Sequence[Soquet], signature: Signature
) -> Dict[str, NDArray[Soquet]]:
    """Split a flat list of soquets into a dictionary corresponding to `signature`."""
    qvars_regs = {}
    base = 0
    for reg in signature:
        assert reg.bitsize == 1
        qvars_regs[reg.name] = np.array(qvars[base : base + reg.total_bits()]).reshape(reg.shape)
        base += reg.total_bits()
    return qvars_regs


def cirq_optree_to_cbloq(
    optree: cirq.OP_TREE,
    *,
    signature: Optional[Signature] = None,
    cirq_quregs: Optional[Dict[str, 'NDArray[cirq.Qid]']] = None,
) -> CompositeBloq:
    """Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.

    Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper.

    If `signature` is not None, the signature of the resultant CompositeBloq is `signature`. For
    multi-dimensional registers and registers with > 1 bitsize, this function automatically
    splits the input soquets into a flat list and joins the output soquets into the correct shape
    to ensure compatibility with the flat API expected by Cirq. When specifying a signature, users
    must also specify the `cirq_quregs` argument, which is a mapping of cirq qubits used in the
    OP-TREE corresponding to the `signature`. If `signature` has registers with entry
        - `Register('x', bitsize=2, shape=(3, 4))` and
        - `Register('y', bitsize=1, shape=(10, 20))`
    then `cirq_quregs` should have one entry corresponding to each register as follows:
        - key='x'; value=`np.array(cirq_qubits_used_in_optree, shape=(3, 4, 2))` and
        - key='y'; value=`np.array(cirq_qubits_used_in_optree, shape=(10, 20, 1))`.

    If `signature` is None, the resultant composite bloq will have one thru-register named "qubits"
    of shape `(n_qubits,)`.
    """
    circuit = cirq.Circuit(optree)
    # "qubits" means cirq qubits | "qvars" means bloq Soquets
    if signature is None:
        assert cirq_quregs is None
        all_qubits = sorted(circuit.all_qubits())
        signature = Signature([Register('qubits', 1, shape=(len(all_qubits),))])
        cirq_quregs = {'qubits': np.array(all_qubits).reshape(len(all_qubits), 1)}

    assert signature is not None and cirq_quregs is not None

    bb, initial_soqs = BloqBuilder.from_signature(signature, add_registers_allowed=False)

    # Magic to make sure signature of the CompositeBloq matches `Signature`.
    qubit_to_qvar = {}
    for reg in signature.lefts():
        assert reg.name in cirq_quregs
        soqs = initial_soqs[reg.name]
        if isinstance(soqs, Soquet):
            soqs = np.asarray(soqs)[np.newaxis, ...]
        if reg.bitsize > 1:
            soqs = np.array([bb.split(soq) for soq in soqs.flatten()])
        soqs = soqs.reshape(reg.shape + (reg.bitsize,))
        assert cirq_quregs[reg.name].shape == soqs.shape
        qubit_to_qvar |= zip(cirq_quregs[reg.name].flatten(), soqs.flatten())

    allocated_qubits = set()
    for op in circuit.all_operations():
        if op.gate is None:
            raise ValueError(f"Only gate operations are supported, not {op}.")

        bloq = CirqGateAsBloq(op.gate)
        for q in op.qubits:
            if q not in qubit_to_qvar:
                qubit_to_qvar[q] = bb.add(Allocate(1))
                allocated_qubits.add(q)

        qvars_in = [qubit_to_qvar[qubit] for qubit in op.qubits]
        qvars_out = bb.add_t(bloq, **_split_qvars_for_regs(qvars_in, bloq.signature))
        qubit_to_qvar |= zip(
            op.qubits, itertools.chain.from_iterable([arr.flatten() for arr in qvars_out])
        )

    for q in allocated_qubits:
        bb.add(Free(1), free=qubit_to_qvar[q])

    qvars = np.array([*qubit_to_qvar.values()])
    final_soqs = {}
    idx = 0
    for reg in signature.rights():
        name = reg.name
        assert name in cirq_quregs
        soqs = qvars[idx : idx + np.product(cirq_quregs[name].shape)]
        idx = idx + np.product(cirq_quregs[name].shape)
        if reg.bitsize > 1:
            # Need to combine the soquets here.
            if len(soqs) == reg.bitsize:
                final_soqs[name] = bb.join(soqs)
            else:
                final_soqs[name] = np.array(
                    [
                        bb.join(soqs[st : st + reg.bitsize])
                        for st in range(0, len(soqs), reg.bitsize)
                    ]
                ).reshape(reg.shape)
        else:
            if len(soqs) == 1 and reg.shape == ():
                final_soqs[name] = soqs[0]
            else:
                final_soqs[name] = soqs.reshape(reg.shape)
    return bb.finalize(**final_soqs)


def _get_in_cirq_quregs(
    binst: BloqInstance, reg: Register, soq_assign: Dict[Soquet, 'NDArray[cirq.Qid]']
) -> 'NDArray[cirq.Qid]':
    """Pluck out the correct values from `soq_assign` for `reg` on `binst`."""
    full_shape = reg.shape + (reg.bitsize,)
    arg = np.empty(full_shape, dtype=object)

    for idx in reg.all_idxs():
        soq = Soquet(binst, reg, idx=idx)
        arg[idx] = soq_assign[soq]

    return arg


def _update_assign_from_cirq_quregs(
    regs: Iterable[Register],
    binst: BloqInstance,
    cirq_quregs: Dict[str, CirqQuregInT],
    soq_assign: Dict[Soquet, CirqQuregT],
) -> None:
    """Update `soq_assign` using `cirq_quregs`.

    This helper function is responsible for error checking. We use `regs` to make sure all the
    keys are present in the vals dictionary. We check the quregs shapes.
    """
    unprocessed_reg_names = set(cirq_quregs.keys())
    for reg in regs:
        try:
            arr = cirq_quregs[reg.name]
        except KeyError:
            raise ValueError(f"{binst} requires an input register named {reg.name}")
        unprocessed_reg_names.remove(reg.name)

        arr = np.asarray(arr)
        full_shape = reg.shape + (reg.bitsize,)
        if arr.shape != full_shape:
            raise ValueError(
                f"Incorrect shape {arr.shape} received for {binst}.{reg.name}. Expected {full_shape}."
            )

        for idx in reg.all_idxs():
            soq = Soquet(binst, reg, idx=idx)
            soq_assign[soq] = arr[idx]

    if unprocessed_reg_names:
        raise ValueError(f"{binst} had extra cirq_quregs: {unprocessed_reg_names}")


def _binst_as_cirq_op(
    binst: BloqInstance,
    pred_cxns: Iterable[Connection],
    soq_assign: Dict[Soquet, NDArray[cirq.Qid]],
    qubit_manager: cirq.QubitManager,
) -> Union[cirq.Operation, None]:
    """Helper function used in `_cbloq_to_cirq_circuit`.

    Args:
        binst: The current BloqInstance on which we wish to call `as_cirq_op`.
        pred_cxns: Predecessor connections for the bloq instance.
        soq_assign: The current assignment from soquets to cirq qubit arrays. This mapping
            is mutated by this function.
        qubit_manager: A `cirq.QubitManager` for allocating `cirq.Qid`s.

    Returns:
        The operation resulting from `binst.bloq.as_cirq_op(...)`.
    """
    # Track inter-Bloq name changes
    for cxn in pred_cxns:
        soq_assign[cxn.right] = soq_assign[cxn.left]
        del soq_assign[cxn.left]

    def _in_vals(reg: Register) -> CirqQuregT:
        # close over `binst` and `soq_assign`.
        return _get_in_cirq_quregs(binst, reg, soq_assign=soq_assign)

    bloq = binst.bloq
    cirq_quregs = {reg.name: _in_vals(reg) for reg in bloq.signature.lefts()}

    op, out_quregs = bloq.as_cirq_op(qubit_manager=qubit_manager, **cirq_quregs)
    _update_assign_from_cirq_quregs(bloq.signature.rights(), binst, out_quregs, soq_assign)
    return op


def decompose_from_cirq_op(bloq: 'Bloq') -> 'CompositeBloq':
    """Returns a CompositeBloq constructed using Cirq operations obtained via `bloq.as_cirq_op`.

    This method first checks whether `bloq.signature` is parameterized. If yes, it raises a
    NotImplementedError. If not, it uses `cirq_optree_to_cbloq` to wrap the operations obtained
    from `bloq.as_cirq_op` into a `CompositeBloq` which has the same signature as `bloq` and returns
    the corresponding `CompositeBloq`.
    """

    if any(
        cirq.is_parameterized(reg.bitsize) or cirq.is_parameterized(reg.side)
        for reg in bloq.signature
    ):
        raise NotImplementedError(f"{bloq} does not support decomposition.")

    cirq_quregs = bloq.signature.get_cirq_quregs()
    cirq_op, cirq_quregs = bloq.as_cirq_op(cirq.ops.SimpleQubitManager(), **cirq_quregs)
    if cirq_op is None or (
        isinstance(cirq_op, cirq.Operation) and isinstance(cirq_op.gate, BloqAsCirqGate)
    ):
        raise NotImplementedError(f"{bloq} does not support decomposition.")

    return cirq_optree_to_cbloq(cirq_op, signature=bloq.signature, cirq_quregs=cirq_quregs)


def _cbloq_to_cirq_circuit(
    signature: Signature,
    cirq_quregs: Dict[str, 'CirqQuregInT'],
    binst_graph: nx.DiGraph,
    qubit_manager: cirq.QubitManager,
) -> Tuple[cirq.FrozenCircuit, Dict[str, 'CirqQuregT']]:
    """Propagate `as_cirq_op` calls through a composite bloq's contents to export a `cirq.Circuit`.

    Args:
        signature: The cbloq's signature for validating inputs and outputs.
        cirq_quregs: Mapping from left register name to Cirq qubit arrays.
        binst_graph: The cbloq's binst graph. This is read only.
        qubit_manager: A `cirq.QubitManager` to allocate new qubits.

    Returns:
        circuit: The cirq.FrozenCircuit version of this composite bloq.
        cirq_quregs: The output mapping from right register names to Cirq qubit arrays.
    """
    soq_assign: Dict[Soquet, CirqQuregT] = {}
    _update_assign_from_cirq_quregs(signature.lefts(), LeftDangle, cirq_quregs, soq_assign)
    moments: List[cirq.Moment] = []
    for binsts in nx.topological_generations(binst_graph):
        moment: List[cirq.Operation] = []

        for binst in binsts:
            if isinstance(binst, DanglingT):
                continue

            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            op = _binst_as_cirq_op(binst, pred_cxns, soq_assign, qubit_manager=qubit_manager)
            if op is not None:
                moment.append(op)
        if moment:
            moments.append(cirq.Moment(moment))

    # Track bloq-to-dangle name changes
    if len(list(signature.rights())) > 0:
        final_preds, _ = _binst_to_cxns(RightDangle, binst_graph=binst_graph)
        for cxn in final_preds:
            soq_assign[cxn.right] = soq_assign[cxn.left]

    # Formulate output with expected API
    def _f_quregs(reg: Register):
        return _get_in_cirq_quregs(RightDangle, reg, soq_assign)

    out_quregs = {reg.name: _f_quregs(reg) for reg in signature.rights()}

    return cirq.FrozenCircuit(moments), out_quregs


class BloqAsCirqGate(cirq_ft.GateWithRegisters):
    """A shim for using bloqs in a Cirq circuit.

    Args:
        bloq: The bloq to wrap.
        reg_to_wires: an optional callable to produce a list of wire symbols for each register
            to match Cirq diagrams.
    """

    def __init__(self, bloq: Bloq, reg_to_wires: Optional[Callable[[Register], List[str]]] = None):
        self._bloq = bloq
        self._legacy_regs, self._compat_name_map = self._init_legacy_regs(bloq)
        self._reg_to_wires = reg_to_wires

    @property
    def bloq(self) -> Bloq:
        """The bloq we're wrapping."""
        return self._bloq

    @property
    def registers(self) -> cirq_ft.Registers:
        """`cirq_ft.GateWithRegisters` registers."""
        return self._legacy_regs

    @staticmethod
    def _init_legacy_regs(bloq: Bloq) -> Tuple[cirq_ft.Registers, Mapping[str, Register]]:
        """Initialize legacy registers.

        We flatten multidimensional registers and annotate non-thru registers with
        modifications to their string name.

        Returns:
            legacy_registers: The flattened, cirq GateWithRegisters-style registers
            compat_name_map: A mapping from the compatability-shim string names of the legacy
                registers back to the original (register, idx) pair.
        """
        legacy_regs: List[cirq_ft.Register] = []
        side_suffixes = {Side.LEFT: '_l', Side.RIGHT: '_r', Side.THRU: ''}
        compat_name_map = {}
        for reg in bloq.signature:
            compat_name = f'{reg.name}{side_suffixes[reg.side]}'
            compat_name_map[compat_name] = reg
            full_shape = reg.shape + (reg.bitsize,)
            legacy_regs.append(cirq_ft.Register(name=compat_name, shape=full_shape))
        return cirq_ft.Registers(legacy_regs), compat_name_map

    @classmethod
    def bloq_on(
        cls, bloq: Bloq, cirq_quregs: Dict[str, 'CirqQuregT'], qubit_manager: cirq.QubitManager
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        """Shim `bloq` into a cirq gate and call it on `cirq_quregs`.

        This is used as a default implementation for `Bloq.as_cirq_op` if a native
        cirq conversion is not specified.

        Args:
            bloq: The bloq to be wrapped with `BloqAsCirqGate`
            cirq_quregs: The cirq qubit registers on which we call the gate.
            qubit_manager: A `cirq.QubitManager` to allocate new qubits.

        Returns:
            op: A cirq operation whose gate is the `BloqAsCirqGate`-wrapped version of `bloq`.
            cirq_quregs: The output cirq qubit registers.
        """
        bloq_quregs: Dict[str, 'CirqQuregT'] = {}
        out_quregs: Dict[str, 'CirqQuregT'] = {}
        for reg in bloq.signature:
            if reg.side is Side.THRU:
                bloq_quregs[reg.name] = cirq_quregs[reg.name]
                out_quregs[reg.name] = cirq_quregs[reg.name]
            elif reg.side is Side.LEFT:
                bloq_quregs[f'{reg.name}_l'] = cirq_quregs[reg.name]
                qubit_manager.qfree(cirq_quregs[reg.name].reshape(-1))
                del cirq_quregs[reg.name]
            elif reg.side is Side.RIGHT:
                new_qubits = qubit_manager.qalloc(reg.total_bits())
                full_shape = reg.shape + (reg.bitsize,)
                out_quregs[reg.name] = np.array(new_qubits).reshape(full_shape)
                bloq_quregs[f'{reg.name}_r'] = out_quregs[reg.name]
        return BloqAsCirqGate(bloq=bloq).on_registers(**bloq_quregs), out_quregs

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Implementation of the GatesWithRegisters decompose method.

        This delegates to `self.bloq.decompose_bloq()` and converts the result to a cirq circuit.

        Args:
            context: `cirq.DecompositionContext` stores options for decomposing gates (eg:
                cirq.QubitManager).
            **quregs: Sequences of cirq qubits as expected for the legacy register shims
            of the bloq's registers.

        Returns:
            A cirq circuit containing the cirq-exported version of the bloq decomposition.
        """
        cbloq = self._bloq.decompose_bloq()

        cirq_quregs: Dict[str, CirqQuregT] = {}
        for compat_name, qubits in quregs.items():
            reg = self._compat_name_map[compat_name]
            cirq_quregs[reg.name] = np.asarray(qubits)

        circuit, _ = cbloq.to_cirq_circuit(qubit_manager=context.qubit_manager, **cirq_quregs)
        return circuit

    def _t_complexity_(self):
        """Delegate to the bloq's t complexity."""
        return self._bloq.t_complexity()

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Draw cirq diagrams.

        By default, we label each qubit with its register name. If `reg_to_wires` was provided
        in the class constructor, we use that to get a list of wire symbols for each register.
        """

        if self._reg_to_wires is not None:
            reg_to_wires = self._reg_to_wires
        else:
            reg_to_wires = lambda reg: [reg.name] * reg.total_bits()

        wire_symbols = []
        for reg in self._bloq.signature:
            symbs = reg_to_wires(reg)
            assert len(symbs) == reg.total_bits()
            wire_symbols.extend(symbs)
        if self._reg_to_wires is None:
            wire_symbols[0] = self._bloq.pretty_name()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __eq__(self, other):
        if not isinstance(other, BloqAsCirqGate):
            return False
        return self.bloq == other.bloq

    def __hash__(self):
        return hash(self.bloq)

    def __str__(self) -> str:
        return f'bloq.{self.bloq}'

    def __repr__(self) -> str:
        return f'BloqAsCirqGate({self.bloq})'
