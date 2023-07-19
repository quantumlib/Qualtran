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

from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cirq
import cirq_ft
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from attrs import frozen
from cirq_ft import Register as LegacyRegister
from cirq_ft import Registers as LegacyRegisters
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

CirqQuregT = NDArray[cirq.Qid]
CirqQuregInT = Union[NDArray[cirq.Qid], Sequence[cirq.Qid]]


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
        return Signature([Register('qubits', 1, shape=(self.n_qubits,))])

    @cached_property
    def n_qubits(self):
        return cirq.num_qubits(self.gate)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        unitary = cirq.unitary(self.gate).reshape((2,) * 2 * self.n_qubits)

        tn.add(
            qtn.Tensor(
                data=unitary,
                inds=outgoing['qubits'].tolist() + incoming['qubits'].tolist(),
                tags=[self.short_name(), tag],
            )
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', qubits: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        assert qubits.shape == (self.n_qubits, 1)
        return self.gate.on(*qubits[:, 0]), {'qubits': qubits}

    def t_complexity(self) -> 'cirq_ft.TComplexity':
        return cirq_ft.t_complexity(self.gate)


def cirq_optree_to_cbloq(
    optree: cirq.OP_TREE, *, signature: Optional[Signature] = None
) -> CompositeBloq:
    """Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.

    Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper.

    If `signature` is not None, the signature of the resultant CompositeBloq is `signature`. For
    multi-dimensional registers and registers with > 1 bitsize, this function automatically
    splits the input soquets into a flat list and joins the output soquets into the correct shape
    to ensure compatibility with the flat API expected by Cirq.

    If `signature` is None, the resultant composite bloq will have one thru-register named "qubits"
    of shape `(n_qubits,)`.
    """
    # "qubits" means cirq qubits | "qvars" means bloq Soquets
    circuit = cirq.Circuit(optree)
    all_qubits = sorted(circuit.all_qubits())
    if signature is None:
        signature = Signature([Register('qubits', 1, shape=(len(all_qubits),))])
    bb, initial_soqs = BloqBuilder.from_signature(signature, add_registers_allowed=False)

    # Magic to make sure signature of the CompositeBloq matches `Signature`.
    qvars = {}
    for reg in signature.lefts():
        soqs = initial_soqs[reg.name]
        if reg.bitsize > 1:
            # Need to split all soquets here.
            if isinstance(soqs, Soquet):
                qvars[reg.name] = bb.split(soqs)
            else:
                qvars[reg.name] = np.concatenate([bb.split(soq) for soq in soqs.reshape(-1)])
        else:
            if isinstance(soqs, Soquet):
                qvars[reg.name] = [soqs]
            else:
                qvars[reg.name] = soqs.reshape(-1)

    qubit_to_qvar = dict(zip(all_qubits, np.concatenate([*qvars.values()])))

    for op in circuit.all_operations():
        if op.gate is None:
            raise ValueError(f"Only gate operations are supported, not {op}.")

        bloq = CirqGateAsBloq(op.gate)
        qvars_for_op = np.array([qubit_to_qvar[qubit] for qubit in op.qubits])
        qvars_for_op_out = bb.add(bloq, qubits=qvars_for_op)
        qubit_to_qvar |= zip(op.qubits, qvars_for_op_out)

    qvar_vals_out = np.array([qubit_to_qvar[qubit] for qubit in all_qubits])

    final_soqs = {}
    idx = 0
    for reg in signature.rights():
        name = reg.name
        soqs = qvar_vals_out[idx : idx + len(qvars[name])]
        idx = idx + len(qvars[name])
        if reg.bitsize > 1:
            # Need to combine the soquets here.
            if len(soqs) == reg.bitsize:
                final_soqs[name] = bb.join(soqs)
            else:
                final_soqs[name] = np.array(
                    bb.join(subsoqs) for subsoqs in soqs[:: reg.bitsize]
                ).reshape(reg.shape)
        else:
            if len(soqs) == 1:
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
            raise ValueError(f"Incorrect shape {arr.shape} received for {binst}.{reg.name}")

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

    return cirq_optree_to_cbloq(cirq_op, signature=bloq.signature)


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
    def registers(self) -> LegacyRegisters:
        """`cirq_ft.GateWithRegisters` registers."""
        return self._legacy_regs

    @staticmethod
    def _init_legacy_regs(
        bloq: Bloq,
    ) -> Tuple[LegacyRegisters, Mapping[str, Tuple[Register, Tuple[int, ...]]]]:
        """Initialize legacy registers.

        We flatten multidimensional registers and annotate non-thru registers with
        modifications to their string name.

        Returns:
            legacy_registers: The flattened, cirq GateWithRegisters-style registers
            compat_name_map: A mapping from the compatability-shim string names of the legacy
                registers back to the original (register, idx) pair.
        """
        legacy_regs: List[LegacyRegister] = []
        side_suffixes = {Side.LEFT: '_l', Side.RIGHT: '_r', Side.THRU: ''}
        compat_name_map = {}
        for reg in bloq.signature:
            if not reg.shape:
                compat_name = f'{reg.name}{side_suffixes[reg.side]}'
                compat_name_map[compat_name] = (reg, ())
                legacy_regs.append(LegacyRegister(name=compat_name, shape=reg.bitsize))
                continue

            for idx in reg.all_idxs():
                idx_str = '_'.join(str(i) for i in idx)
                compat_name = f'{reg.name}{side_suffixes[reg.side]}_{idx_str}'
                compat_name_map[compat_name] = (reg, idx)
                legacy_regs.append(LegacyRegister(name=compat_name, shape=reg.bitsize))

        return LegacyRegisters(legacy_regs), compat_name_map

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
        flat_qubits: List[cirq.Qid] = []
        out_quregs: Dict[str, 'CirqQuregT'] = {}
        for reg in bloq.signature:
            if reg.side is Side.THRU:
                for i, q in enumerate(cirq_quregs[reg.name].reshape(-1)):
                    flat_qubits.append(q)
                out_quregs[reg.name] = cirq_quregs[reg.name]
            elif reg.side is Side.LEFT:
                for i, q in enumerate(cirq_quregs[reg.name].reshape(-1)):
                    flat_qubits.append(q)
                qubit_manager.qfree(cirq_quregs[reg.name].reshape(-1))
                del cirq_quregs[reg.name]
            elif reg.side is Side.RIGHT:
                new_qubits = qubit_manager.qalloc(reg.total_bits())
                flat_qubits.extend(new_qubits)
                out_quregs[reg.name] = np.array(new_qubits).reshape(reg.shape + (reg.bitsize,))

        return BloqAsCirqGate(bloq=bloq).on(*flat_qubits), out_quregs

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **qubit_regs: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        """Implementation of the GatesWithRegisters decompose method.

        This delegates to `self.bloq.decompose_bloq()` and converts the result to a cirq circuit.

        Args:
            context: `cirq.DecompositionContext` stores options for decomposing gates (eg:
                cirq.QubitManager).
            **qubit_regs: Sequences of cirq qubits as expected for the legacy register shims
            of the bloq's registers.

        Returns:
            A cirq circuit containing the cirq-exported version of the bloq decomposition.
        """
        cbloq = self._bloq.decompose_bloq()

        # Initialize shapely qubit registers to pass to bloqs infrastructure
        cirq_quregs: Dict[str, CirqQuregT] = {}
        for reg in self._bloq.signature:
            if reg.shape:
                shape = reg.shape + (reg.bitsize,)
                cirq_quregs[reg.name] = np.empty(shape, dtype=object)

        # Shapefy the provided cirq qubits
        for compat_name, qubits in qubit_regs.items():
            reg, idx = self._compat_name_map[compat_name]
            if idx == ():
                cirq_quregs[reg.name] = np.asarray(qubits)
            else:
                cirq_quregs[reg.name][idx] = np.asarray(qubits)

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
