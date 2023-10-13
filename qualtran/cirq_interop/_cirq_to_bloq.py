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

"""Cirq gates/circuits to Qualtran Bloqs conversion."""
import itertools
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import cirq
import numpy as np
import quimb.tensor as qtn
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, CompositeBloq, Register, Side, Signature, Soquet, SoquetT
from qualtran._infra.gate_with_registers import split_qubits
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager
from qualtran.cirq_interop.t_complexity_protocol import t_complexity, TComplexity

if TYPE_CHECKING:
    from qualtran.drawing import WireSymbol

CirqQuregT = NDArray[cirq.Qid]
CirqQuregInT = Union[NDArray[cirq.Qid], Sequence[cirq.Qid]]


def get_cirq_quregs(signature: Signature, qm: InteropQubitManager):
    ret = signature.get_cirq_quregs()
    qm.manage_qubits(itertools.chain.from_iterable(qreg.flatten() for qreg in ret.values()))
    return ret


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
        if isinstance(self.gate, Bloq):
            return self.gate.signature
        import cirq_ft

        if isinstance(self.gate, cirq_ft.GateWithRegisters):
            # TODO(gh/Qualtran/issues/398): Remove once `cirq_ft.GateWithRegisters` is deprecated.
            return Signature(
                [
                    Register(reg.name, reg.bitsize, reg.shape, Side(reg.side.value))
                    for reg in self.gate.signature
                ]
            )
        return Signature([Register('qubits', shape=cirq.num_qubits(self.gate), bitsize=1)])

    def decompose_bloq(self) -> 'CompositeBloq':
        return decompose_from_cirq_op(self, decompose_once=True)

    def add_my_tensors(
        self,
        tn: qtn.TensorNetwork,
        tag: Any,
        *,
        incoming: Dict[str, 'SoquetT'],
        outgoing: Dict[str, 'SoquetT'],
    ):
        _add_my_tensors_from_gate(
            self.gate,
            self.signature,
            self.short_name(),
            tn=tn,
            tag=tag,
            incoming=incoming,
            outgoing=outgoing,
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **cirq_quregs: 'CirqQuregT'
    ) -> Tuple['cirq.Operation', Dict[str, 'CirqQuregT']]:
        import cirq_ft

        from qualtran import GateWithRegisters
        from qualtran.cirq_interop._bloq_to_cirq import _construct_op_from_gate

        if not isinstance(self.gate, (cirq_ft.GateWithRegisters, GateWithRegisters)):
            return self.gate.on(*cirq_quregs['qubits'].flatten()), cirq_quregs
        return _construct_op_from_gate(
            self.gate,
            in_quregs={k: np.array(v) for k, v in cirq_quregs.items()},
            qubit_manager=qubit_manager,
        )

    def t_complexity(self) -> 'TComplexity':
        return t_complexity(self.gate)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        return _wire_symbol_from_gate(self.gate, self.signature, soq)


def _wire_symbol_from_gate(gate: cirq.Gate, signature: Signature, soq: 'Soquet') -> 'WireSymbol':
    from qualtran.drawing import directional_text_box

    wire_symbols = cirq.circuit_diagram_info(gate).wire_symbols
    begin = 0
    symbol: str = soq.pretty()
    for reg in signature:
        finish = begin + int(np.prod(reg.shape))
        if reg == soq.reg:
            symbol = np.array(wire_symbols[begin:finish]).reshape(reg.shape)[soq.idx]
        begin = finish
    return directional_text_box(text=symbol, side=soq.reg.side)


def _add_my_tensors_from_gate(
    gate: cirq.Gate,
    signature: Signature,
    short_name: str,
    tn: qtn.TensorNetwork,
    tag: Any,
    *,
    incoming: Dict[str, 'SoquetT'],
    outgoing: Dict[str, 'SoquetT'],
):
    if not cirq.has_unitary(gate):
        raise NotImplementedError(
            f"CirqGateAsBloq.add_my_tensors is currently supported only for unitary gates. "
            f"Found {gate}."
        )
    unitary_shape = []
    reg_to_idx = defaultdict(list)
    for reg in signature:
        start = len(unitary_shape)
        for i in range(int(np.prod(reg.shape))):
            reg_to_idx[reg.name].append(start + i)
            unitary_shape.append(2**reg.bitsize)

    unitary_shape = (*unitary_shape, *unitary_shape)
    unitary = cirq.unitary(gate).reshape(unitary_shape)
    idx: List[Union[int, slice]] = [slice(x) for x in unitary_shape]
    n = len(unitary_shape) // 2
    for reg in signature:
        if reg.side == Side.LEFT:
            for i in reg_to_idx[reg.name]:
                # LEFT register ends, extract right subspace that's equivalent to 0.
                idx[i] = 0
        if reg.side == Side.RIGHT:
            for i in reg_to_idx[reg.name]:
                # Right register begins, extract the left subspace that's equivalent to 0.
                idx[i + n] = 0
    unitary = unitary[tuple(idx)]
    new_shape = tuple(
        [
            *itertools.chain.from_iterable(
                (2**reg.bitsize,) * int(np.prod(reg.shape))
                for reg in [*signature.rights(), *signature.lefts()]
            )
        ]
    )
    assert unitary.shape == new_shape
    incoming_list = [
        *itertools.chain.from_iterable(
            [np.array(incoming[reg.name]).flatten() for reg in signature.lefts()]
        )
    ]
    outgoing_list = [
        *itertools.chain.from_iterable(
            [np.array(outgoing[reg.name]).flatten() for reg in signature.rights()]
        )
    ]
    tn.add(qtn.Tensor(data=unitary, inds=outgoing_list + incoming_list, tags=[short_name, tag]))


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

        bloq = op.gate if isinstance(op.gate, Bloq) else CirqGateAsBloq(op.gate)
        # 3.1 Find input / output registers.
        all_op_quregs: Dict[str, NDArray[_QReg]] = {
            k: np.apply_along_axis(_QReg, -1, v)
            for k, v in split_qubits(bloq.signature, op.qubits).items()
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


def decompose_from_cirq_op(bloq: 'Bloq', *, decompose_once: bool = False) -> 'CompositeBloq':
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

    qubit_manager = InteropQubitManager()
    in_quregs = get_cirq_quregs(bloq.signature, qubit_manager)
    cirq_op, out_quregs = bloq.as_cirq_op(qubit_manager, **in_quregs)
    context = cirq.DecompositionContext(qubit_manager=qubit_manager)
    decomposed_optree = (
        cirq.decompose_once(cirq_op, context=context, default=None) if decompose_once else cirq_op
    )

    if decomposed_optree is None:
        raise NotImplementedError(f"{bloq} does not support decomposition.")

    return cirq_optree_to_cbloq(
        decomposed_optree, signature=bloq.signature, in_quregs=in_quregs, out_quregs=out_quregs
    )
