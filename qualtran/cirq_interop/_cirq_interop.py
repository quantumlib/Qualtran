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

"""Bi-directional interop between Qualtran & Cirq using Cirq-FT."""
import itertools
from functools import cached_property
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cirq
import cirq_ft
import networkx as nx
import numpy as np
import quimb.tensor as qtn
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    Connection,
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


def signature_from_cirq_registers(registers: Iterable[cirq_ft.Register]) -> 'Signature':
    return Signature(
        [
            Register(reg.name, bitsize=reg.bitsize, shape=reg.shape, side=Side(reg.side.value))
            for reg in registers
        ]
    )


# Part-I: Cirq to Bloq conversion.


@frozen
class CirqGateAsBloq(Bloq):
    """A Bloq wrapper around a `cirq.Gate`, preserving signature if gate is a `GateWithRegisters`."""

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
    def cirq_registers(self) -> cirq_ft.Signature:
        if isinstance(self.gate, cirq_ft.GateWithRegisters):
            return self.gate.signature
        else:
            return cirq_ft.Signature(
                [cirq_ft.Register('qubits', shape=(cirq.num_qubits(self.gate),), bitsize=1)]
            )

    def decompose_bloq(self) -> 'CompositeBloq':
        in_quregs = self.signature.get_cirq_quregs()
        qubit_manager = cirq.ops.SimpleQubitManager()
        cirq_op, out_quregs = self.as_cirq_op(qubit_manager, **in_quregs)
        context = cirq.DecompositionContext(qubit_manager=qubit_manager)
        decomposed_optree = cirq.decompose_once(cirq_op, context=context, default=None)
        if decomposed_optree is None:
            raise NotImplementedError(f"{self} does not support decomposition.")
        return cirq_optree_to_cbloq(
            decomposed_optree, signature=self.signature, in_quregs=in_quregs, out_quregs=out_quregs
        )

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        new_shape = [
            *itertools.chain.from_iterable(
                (2**reg.bitsize,) * int(np.prod(reg.shape))
                for reg in [*self.signature.rights(), *self.signature.lefts()]
            )
        ]
        unitary = cirq.unitary(self.gate).reshape(new_shape)
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
        if not isinstance(self.gate, cirq_ft.GateWithRegisters):
            return self.gate.on(*cirq_quregs['qubits'].flatten()), cirq_quregs
        return _construct_op_from_gate(
            self.gate,
            in_quregs={k: np.array(v) for k, v in cirq_quregs.items()},
            qubit_manager=qubit_manager,
        )

    def t_complexity(self) -> 'cirq_ft.TComplexity':
        return cirq_ft.t_complexity(self.gate)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        from qualtran.drawing import directional_text_box

        wire_symbols = cirq.circuit_diagram_info(self.gate).wire_symbols
        begin = 0
        symbol: str = soq.pretty()
        for reg in self.signature:
            finish = begin + int(np.prod(reg.shape))
            if reg == soq.reg:
                symbol = np.array(wire_symbols[begin:finish]).reshape(reg.shape)[soq.idx]
            begin = finish
        return directional_text_box(text=symbol, side=soq.reg.side)


@frozen
class _QReg:
    """Used as a container for qubits that form a `cirq_ft.Register` of a given bitsize.

    Each instance of `_QReg` would correspond to a `Soquet` in Bloqs and represents an opaque collection
    of qubits that together form a quantum register.
    """

    qubits: Tuple[cirq.Qid, ...] = field(
        converter=lambda v: (v,) if isinstance(v, cirq.Qid) else tuple(v)
    )


def _ensure_in_reg_exists(
    bb: BloqBuilder, in_reg: _QReg, qreg_to_qvar: Dict[_QReg, Soquet]
) -> None:
    """Takes care of qubit allocations, split and joins to ensure `qreg_to_qvar[in_reg]` exists."""
    all_mapped_qubits = {q for qreg in qreg_to_qvar for q in qreg.qubits}
    qubits_to_allocate: List[cirq.Qid] = [q for q in in_reg.qubits if q not in all_mapped_qubits]
    if qubits_to_allocate:
        qreg_to_qvar[_QReg(qubits_to_allocate)] = bb.allocate(len(qubits_to_allocate))

    if in_reg in qreg_to_qvar:
        # This is the easy case when no split / joins are needed.
        return

    # a. Split all registers containing at-least one qubit corresponding to `in_reg`.
    in_reg_qubits = set(in_reg.qubits)

    new_qreg_to_qvar: Dict[_QReg, Soquet] = {}
    for qreg, soq in qreg_to_qvar.items():
        if len(qreg.qubits) > 1 and any(q in qreg.qubits for q in in_reg_qubits):
            new_qreg_to_qvar |= {_QReg(q): s for q, s in zip(qreg.qubits, bb.split(soq=soq))}
        else:
            new_qreg_to_qvar[qreg] = soq
    qreg_to_qvar.clear()

    # b. Join all 1-bit registers, corresponding to individual qubits, that make up `in_reg`.
    soqs_to_join = []
    for qreg, soq in new_qreg_to_qvar.items():
        if len(in_reg_qubits) > 1 and qreg.qubits and qreg.qubits[0] in in_reg_qubits:
            assert len(qreg.qubits) == 1, "Individual qubits should have been split by now."
            soqs_to_join.append(soq)
        else:
            qreg_to_qvar[qreg] = soq
    if soqs_to_join:
        qreg_to_qvar[in_reg] = bb.join(np.array(soqs_to_join))


def _gather_input_soqs(
    bb: BloqBuilder, op_quregs: Dict[str, NDArray[_QReg]], qreg_to_qvar: Dict[_QReg, Soquet]
) -> Dict[str, NDArray[Soquet]]:
    qvars_in: Dict[str, NDArray[Soquet]] = {}
    for reg_name, quregs in op_quregs.items():
        flat_soqs: List[Soquet] = []
        for qureg in quregs.flatten():
            _ensure_in_reg_exists(bb, qureg, qreg_to_qvar)
            flat_soqs.append(qreg_to_qvar[qureg])
        qvars_in[reg_name] = np.array(flat_soqs).reshape(quregs.shape)
    return qvars_in


def cirq_optree_to_cbloq(
    optree: cirq.OP_TREE,
    *,
    signature: Optional[Signature] = None,
    in_quregs: Optional[Dict[str, 'CirqQuregT']] = None,
    out_quregs: Optional[Dict[str, 'CirqQuregT']] = None,
) -> CompositeBloq:
    """Convert a Cirq OP-TREE into a `CompositeBloq` with signature `signature`.

     Each `cirq.Operation` will be wrapped into a `CirqGateAsBloq` wrapper.
     The signature of the resultant CompositeBloq is `signature`, if provided. Otherwise, use
     one thru-register named "qubits" of shape `(n_qubits,)`.

     For multi-dimensional registers and registers with bitsize>1, this function automatically
     splits the input soquets and joins the output soquets to ensure compatibility with the
     flat-list-of-qubits expected by Cirq.

     When specifying a signature, users must also specify the `in_quregs` & `out_quregs` arguments,
     which are mappings of cirq qubits used in the OP-TREE corresponding to the `LEFT` & `RIGHT`
     registers in `signature`. If `signature` has registers with entry

        - `Register('x', bitsize=2, shape=(3, 4), side=Side.THRU)`
        - `Register('y', bitsize=1, shape=(10, 20), side=Side.LEFT)`
        - `Register('z', bitsize=1, shape=(10, 20), side=Side.RIGHT)`

    then `in_quregs` should have one entry corresponding to registers `x` and `y` as follows:

        - key='x'; value=`np.array(cirq_qubits_used_for_x, shape=(3, 4, 2))` and
        - key='y'; value=`np.array(cirq_qubits_used_for_y, shape=(10, 20, 1))`.
    and `out_quregs` should have one entry corresponding to registers `x` and `z` as follows:

        - key='x'; value=`np.array(cirq_qubits_used_for_x, shape=(3, 4, 2))` and
        - key='z'; value=`np.array(cirq_qubits_used_for_z, shape=(10, 20, 1))`.

    Any qubit in `optree` which is not part of `in_quregs` and `out_quregs` is considered to be
    allocated & deallocated inside the CompositeBloq and does not show up in it's signature.
    """
    circuit = cirq.Circuit(optree)
    if signature is None:
        if in_quregs is not None or out_quregs is not None:
            raise ValueError("`in_quregs` / `out_quregs` requires specifying `signature`.")
        all_qubits = sorted(circuit.all_qubits())
        signature = Signature([Register('qubits', 1, shape=(len(all_qubits),))])
        in_quregs = out_quregs = {'qubits': np.array(all_qubits).reshape(len(all_qubits), 1)}
    elif in_quregs is None or out_quregs is None:
        raise ValueError("`signature` requires specifying both `in_quregs` and `out_quregs`.")

    in_quregs = {k: np.apply_along_axis(_QReg, -1, v) for k, v in in_quregs.items()}
    out_quregs = {k: np.apply_along_axis(_QReg, -1, v) for k, v in out_quregs.items()}

    bb, initial_soqs = BloqBuilder.from_signature(signature, add_registers_allowed=False)

    # 1. Compute qreg_to_qvar for input qubits in the LEFT signature.
    qreg_to_qvar: Dict[_QReg, Soquet] = {}
    for reg in signature.lefts():
        if reg.name not in in_quregs:
            raise ValueError(f"Register {reg.name} from signature must be present in in_quregs.")
        soqs = initial_soqs[reg.name]
        if isinstance(soqs, Soquet):
            soqs = np.array(soqs)
        if in_quregs[reg.name].shape != soqs.shape:
            raise ValueError(
                f"Shape {in_quregs[reg.name].shape} of cirq register "
                f"{reg.name} should be {soqs.shape}."
            )
        qreg_to_qvar |= zip(in_quregs[reg.name].flatten(), soqs.flatten())

    # 2. Add each operation to the composite Bloq.
    for op in circuit.all_operations():
        if op.gate is None:
            raise ValueError(f"Only gate operations are supported, not {op}.")

        bloq = CirqGateAsBloq(op.gate)
        # 3.1 Find input / output registers.
        all_op_quregs: Dict[str, NDArray[_QReg]] = {
            k: np.apply_along_axis(_QReg, -1, v)
            for k, v in cirq_ft.infra.split_qubits(bloq.cirq_registers, op.qubits).items()
        }
        in_op_quregs: Dict[str, NDArray[_QReg]] = {
            reg.name: all_op_quregs[reg.name] for reg in bloq.signature.lefts()
        }
        # 3.2 Find input Soquets, by potentially allocating new Bloq registers corresponding to
        # input Cirq `in_quregs` and updating the `qreg_to_qvar` mapping.
        qvars_in = _gather_input_soqs(bb, in_op_quregs, qreg_to_qvar)

        # 3.3 Add Bloq to the `CompositeBloq` compute graph and get corresponding output Soquets.
        qvars_out = bb.add_d(bloq, **qvars_in)

        # 3.4 Update `qreg_to_qvar` mapping using output soquets `qvars_out`.
        for reg in bloq.signature:
            # all_op_quregs should exist for both LEFT & RIGHT registers.
            assert reg.name in all_op_quregs
            quregs = all_op_quregs[reg.name]
            if reg.side == Side.LEFT:
                # This register got de-allocated, update the `qreg_to_qvar` mapping.
                for q in quregs.flatten():
                    _ = qreg_to_qvar.pop(q)
            else:
                assert quregs.shape == np.array(qvars_out[reg.name]).shape
                qreg_to_qvar |= zip(quregs.flatten(), np.array(qvars_out[reg.name]).flatten())

    # 4. Combine Soquets to match the right signature.
    final_soqs_dict = _gather_input_soqs(
        bb, {reg.name: out_quregs[reg.name] for reg in signature.rights()}, qreg_to_qvar
    )
    final_soqs_set = set(soq for soqs in final_soqs_dict.values() for soq in soqs.flatten())
    # 5. Free all dangling Soquets which are not part of the final soquets set.
    for qvar in qreg_to_qvar.values():
        if qvar not in final_soqs_set:
            bb.free(qvar)
    return bb.finalize(**final_soqs_dict)


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

    in_quregs = bloq.signature.get_cirq_quregs()
    cirq_op, out_quregs = bloq.as_cirq_op(cirq.ops.SimpleQubitManager(), **in_quregs)
    if cirq_op is None or (
        isinstance(cirq_op, cirq.Operation) and isinstance(cirq_op.gate, BloqAsCirqGate)
    ):
        raise NotImplementedError(f"{bloq} does not support decomposition.")

    return cirq_optree_to_cbloq(
        cirq_op, signature=bloq.signature, in_quregs=in_quregs, out_quregs=out_quregs
    )


# Part-II: Bloq to Cirq conversion.


def _track_soq_name_changes(cxns: Iterable[Connection], qvar_to_qreg: Dict[Soquet, _QReg]):
    """Track inter-Bloq name changes across the two ends of a connection."""
    for cxn in cxns:
        qvar_to_qreg[cxn.right] = qvar_to_qreg[cxn.left]
        del qvar_to_qreg[cxn.left]


def _bloq_to_cirq_op(
    bloq: Bloq,
    pred_cxns: Iterable[Connection],
    succ_cxns: Iterable[Connection],
    qvar_to_qreg: Dict[Soquet, _QReg],
    qubit_manager: cirq.QubitManager,
) -> cirq.Operation:
    _track_soq_name_changes(pred_cxns, qvar_to_qreg)
    in_quregs: Dict[str, CirqQuregT] = {
        reg.name: np.empty((*reg.shape, reg.bitsize), dtype=object)
        for reg in bloq.signature.lefts()
    }
    # Construct the cirq qubit registers using input / output connections.
    # 1. All input Soquets should already have the correct mapping in `qvar_to_qreg`.
    for cxn in pred_cxns:
        soq = cxn.right
        assert soq in qvar_to_qreg, f"{soq=} should exist in {qvar_to_qreg=}."
        in_quregs[soq.reg.name][soq.idx] = qvar_to_qreg[soq].qubits
        if soq.reg.side == Side.LEFT:
            # Remove soquets for LEFT registers from qvar_to_qreg mapping.
            del qvar_to_qreg[soq]

    op, out_quregs = bloq.as_cirq_op(qubit_manager=qubit_manager, **in_quregs)

    # 2. Update the mappings based on output soquets and `out_quregs`.
    for cxn in succ_cxns:
        soq = cxn.left
        assert soq.reg.name in out_quregs, f"{soq=} should exist in {out_quregs=}."
        if soq.reg.side == Side.RIGHT:
            qvar_to_qreg[soq] = _QReg(out_quregs[soq.reg.name][soq.idx])
    return op


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
    cirq_quregs = {k: np.apply_along_axis(_QReg, -1, v) for k, v in cirq_quregs.items()}
    qvar_to_qreg: Dict[Soquet, _QReg] = {
        Soquet(LeftDangle, idx=idx, reg=reg): cirq_quregs[reg.name][idx]
        for reg in signature.lefts()
        for idx in reg.all_idxs()
    }
    moments: List[cirq.Moment] = []
    for binsts in nx.topological_generations(binst_graph):
        moment: List[cirq.Operation] = []

        for binst in binsts:
            if binst == LeftDangle:
                continue
            pred_cxns, succ_cxns = _binst_to_cxns(binst, binst_graph=binst_graph)
            if binst == RightDangle:
                _track_soq_name_changes(pred_cxns, qvar_to_qreg)
                continue

            op = _bloq_to_cirq_op(binst.bloq, pred_cxns, succ_cxns, qvar_to_qreg, qubit_manager)
            if op is not None:
                moment.append(op)
        if moment:
            moments.append(cirq.Moment(moment))

    # Find output Cirq quregs using `qvar_to_qreg` mapping for registers in `signature.rights()`.
    def _f_quregs(reg: Register) -> CirqQuregT:
        ret = np.empty(reg.shape + (reg.bitsize,), dtype=object)
        for idx in reg.all_idxs():
            soq = Soquet(RightDangle, idx=idx, reg=reg)
            ret[idx] = qvar_to_qreg[soq].qubits
        return ret

    out_quregs = {reg.name: _f_quregs(reg) for reg in signature.rights()}

    return cirq.FrozenCircuit(moments), out_quregs


def _construct_op_from_gate(
    gate: cirq_ft.GateWithRegisters,
    in_quregs: Dict[str, 'CirqQuregT'],
    qubit_manager: cirq.QubitManager,
) -> Tuple[cirq.Operation, Dict[str, 'CirqQuregT']]:
    """Allocates / Deallocates qubits for RIGHT / LEFT only registers to construct a Cirq operation

    Args:
        gate: A `cirq_ft.GateWithRegisters` which specifies a signature.
        in_quregs: Mapping from LEFT register names of `gate` and corresponding cirq qubits.
        qubit_manager: For allocating / deallocating qubits for RIGHT / LEFT only registers.

    Returns:
        A cirq operation constructed using `gate` and a mapping from RIGHT register names to
        corresponding Cirq qubits.
    """
    all_quregs: Dict[str, 'CirqQuregT'] = {}
    out_quregs: Dict[str, 'CirqQuregT'] = {}
    for reg in gate.signature:
        full_shape = reg.shape + (reg.bitsize,)
        if reg.side & cirq_ft.infra.Side.LEFT:
            if reg.name not in in_quregs or in_quregs[reg.name].shape != full_shape:
                # Left registers should exist as input to `as_cirq_op`.
                raise ValueError(f'Compatible {reg=} must exist in {in_quregs=}')
            all_quregs[reg.name] = in_quregs[reg.name]
        if reg.side == cirq_ft.infra.Side.RIGHT:
            # Right only registers will get allocated as part of `as_cirq_op`.
            if reg.name in in_quregs:
                raise ValueError(f"RIGHT register {reg=} shouldn't exist in {in_quregs=}.")
            all_quregs[reg.name] = np.array(qubit_manager.qalloc(reg.total_bits())).reshape(
                full_shape
            )
        if reg.side == cirq_ft.infra.Side.LEFT:
            # LEFT only registers should be de-allocated and not be part of output.
            qubit_manager.qfree(in_quregs[reg.name].flatten())

        if reg.side & cirq_ft.infra.Side.RIGHT:
            # Right registers should be part of the output.
            out_quregs[reg.name] = all_quregs[reg.name]
    return gate.on_registers(**all_quregs), out_quregs


class BloqAsCirqGate(cirq_ft.GateWithRegisters):
    """A shim for using bloqs in a Cirq circuit.

    Args:
        bloq: The bloq to wrap.
        reg_to_wires: an optional callable to produce a list of wire symbols for each register
            to match Cirq diagrams.
    """

    def __init__(self, bloq: Bloq, reg_to_wires: Optional[Callable[[Register], List[str]]] = None):
        for _, regs in bloq.signature.groups():
            if len(regs) > 1:
                raise ValueError(
                    f"Automated cirq conversion doesn't support multiple registers with same name."
                    f" Found {regs}\n. Please override `bloq.as_cirq_op` for `{bloq=}` instead."
                )
        self._bloq = bloq
        self._reg_to_wires = reg_to_wires

    @property
    def bloq(self) -> Bloq:
        """The bloq we're wrapping."""
        return self._bloq

    @cached_property
    def signature(self) -> cirq_ft.Signature:
        """`cirq_ft.GateWithRegisters` registers."""
        legacy_regs: List[cirq_ft.Register] = []
        for reg in self.bloq.signature:
            legacy_regs.append(
                cirq_ft.Register(
                    name=reg.name,
                    shape=reg.shape,
                    bitsize=reg.bitsize,
                    side=cirq_ft.infra.Side(reg.side.value),
                )
            )
        return cirq_ft.Signature(legacy_regs)

    @classmethod
    def bloq_on(
        cls, bloq: Bloq, cirq_quregs: Dict[str, 'CirqQuregT'], qubit_manager: cirq.QubitManager
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        """Shim `bloq` into a cirq gate and call it on `cirq_quregs`.

        This is used as a default implementation for `Bloq.as_cirq_op` if a native
        cirq conversion is not specified.

        Args:
            bloq: The bloq to be wrapped with `BloqAsCirqGate`
            cirq_quregs: The cirq qubit registers on which we call the gate. Should correspond to
                registers in `self.bloq.signature.lefts()`.
            qubit_manager: A `cirq.QubitManager` to allocate new qubits.

        Returns:
            op: A cirq operation whose gate is the `BloqAsCirqGate`-wrapped version of `bloq`.
            cirq_quregs: The output cirq qubit registers.
        """
        return _construct_op_from_gate(
            BloqAsCirqGate(bloq=bloq), in_quregs=cirq_quregs, qubit_manager=qubit_manager
        )

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs: CirqQuregT
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
        circuit, out_quregs = cbloq.to_cirq_circuit(qubit_manager=context.qubit_manager, **quregs)
        qubit_map = {q: q for q in circuit.all_qubits()}
        for reg in self.bloq.signature.rights():
            if reg.side == Side.RIGHT:
                # Right only registers can get mapped to newly allocated output qubits in `out_regs`.
                # Map them back to the original system qubits and deallocate newly allocated qubits.
                assert reg.name in quregs and reg.name in out_quregs
                assert quregs[reg.name].shape == out_quregs[reg.name].shape
                context.qubit_manager.qfree([q for q in out_quregs[reg.name].flatten()])
                qubit_map |= zip(out_quregs[reg.name].flatten(), quregs[reg.name].flatten())
        return circuit.unfreeze(copy=False).transform_qubits(qubit_map)

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
