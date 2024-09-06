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
import abc
import itertools
import numbers
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, TypeVar, Union

import cirq
import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    ConnectionT,
    CtrlSpec,
    DecomposeNotImplementedError,
    DecomposeTypeError,
    GateWithRegisters,
    QAny,
    QBit,
    QDType,
    Register,
    Side,
    Signature,
    Soquet,
)
from qualtran._infra.gate_with_registers import (
    _get_all_and_output_quregs_from_input,
    get_named_qubits,
    split_qubits,
)
from qualtran.cirq_interop._interop_qubit_manager import InteropQubitManager
from qualtran.cirq_interop.t_complexity_protocol import _from_directly_countable_cirq, TComplexity
from qualtran.resource_counting import CostKey, GateCounts, QECGatesCost

if TYPE_CHECKING:
    import quimb.tensor as qtn

    from qualtran.drawing import WireSymbol


# numpy subtypes must be np.generic
# However, this denotes a numpy array of type cirq.Qid
_QidType = TypeVar('_QidType', bound=np.generic)

CirqQuregT = NDArray[_QidType]
CirqQuregInT = Union[NDArray[_QidType], Sequence[cirq.Qid]]


def _get_cirq_quregs(signature: Signature, qm: InteropQubitManager):
    ret = get_named_qubits(signature.lefts())
    qm.manage_qubits(itertools.chain.from_iterable(qreg.flatten() for qreg in ret.values()))
    return ret


class CirqGateAsBloqBase(GateWithRegisters, metaclass=abc.ABCMeta):
    """A Bloq wrapper around a `cirq.Gate`"""

    @property
    @abc.abstractmethod
    def cirq_gate(self) -> cirq.Gate:
        ...

    @cached_property
    def signature(self) -> 'Signature':
        if isinstance(self.cirq_gate, Bloq):
            return self.cirq_gate.signature
        nqubits = cirq.num_qubits(self.cirq_gate)
        return (
            Signature([Register('q', QBit(), shape=nqubits)])
            if nqubits > 1
            else Signature.build(q=nqubits)
        )

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: CirqQuregT
    ) -> cirq.OP_TREE:
        op = (
            self.cirq_gate.on_registers(**quregs)
            if isinstance(self.cirq_gate, GateWithRegisters)
            else self.cirq_gate.on(*quregs.get('q', np.array(())).flatten())
        )
        try:
            return cirq.decompose_once(op)
        except TypeError as e:
            raise DecomposeNotImplementedError(f"{self} does not declare a decomposition.") from e

    def my_tensors(
        self, incoming: Dict[str, 'ConnectionT'], outgoing: Dict[str, 'ConnectionT']
    ) -> List['qtn.Tensor']:
        return _my_tensors_from_gate(
            self.cirq_gate, self.signature, incoming=incoming, outgoing=outgoing
        )

    def as_cirq_op(
        self, qubit_manager: 'cirq.QubitManager', **in_quregs: 'CirqQuregT'
    ) -> Tuple[Union['cirq.Operation', None], Dict[str, 'CirqQuregT']]:
        if isinstance(self.cirq_gate, GateWithRegisters):
            return self.cirq_gate.as_cirq_op(qubit_manager, **in_quregs)
        qubits = in_quregs.get('q', np.array([])).flatten()
        return self.cirq_gate.on(*qubits), in_quregs

    # Delegate all cirq-style protocols to underlying gate
    def _unitary_(self):
        return cirq.unitary(self.cirq_gate, default=None)

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> Optional[cirq.CircuitDiagramInfo]:
        return cirq.circuit_diagram_info(self.cirq_gate, default=None)

    def __str__(self):
        return str(self.cirq_gate)

    def __pow__(self, power):
        return CirqGateAsBloq(gate=cirq.pow(self.cirq_gate, power))

    def adjoint(self) -> 'Bloq':
        return CirqGateAsBloq(gate=cirq.inverse(self.cirq_gate))


@frozen
class CirqGateAsBloq(CirqGateAsBloqBase):
    gate: cirq.Gate

    def __str__(self) -> str:
        g = min(self.cirq_gate.__class__.__name__, str(self.cirq_gate), key=len)
        return f'cirq.{g}'

    @property
    def cirq_gate(self) -> cirq.Gate:
        return self.gate

    def _t_complexity_(self) -> 'TComplexity':
        t_count = _from_directly_countable_cirq(self.cirq_gate)
        if t_count is None:
            raise ValueError(f"Cirq gate must be directly countable, not {self.cirq_gate}")
        return t_count

    def my_static_costs(self, cost_key: 'CostKey'):
        if isinstance(cost_key, QECGatesCost):
            t_count = _from_directly_countable_cirq(self.cirq_gate)
            if t_count is None:
                raise ValueError(f"Cirq gate must be directly countable, not {self.cirq_gate}")
            return GateCounts(t=t_count.t, rotation=t_count.rotations, clifford=t_count.clifford)


def _cirq_wire_symbol_to_qualtran_wire_symbol(symbol: str, side: Side) -> 'WireSymbol':
    from qualtran.drawing import Circle, directional_text_box, ModPlus

    if symbol == "@":
        return Circle(filled=True)
    if symbol == "@(0)":
        return Circle(filled=False)
    if symbol == "X":
        return ModPlus()
    return directional_text_box(symbol, side=side)


def _wire_symbol_from_gate(
    gate: cirq.Gate, signature: Signature, wire_reg: Register, idx: Tuple[int, ...] = tuple()
) -> 'WireSymbol':
    wire_symbols = cirq.circuit_diagram_info(gate).wire_symbols
    begin = 0
    if len(idx) > 0:
        symbol = f'{wire_reg.name}[{", ".join(str(i) for i in idx)}]'
    else:
        symbol = wire_reg.name
    for reg in signature:
        reg_size = int(np.prod(reg.shape))
        finish = begin + reg.bitsize * int(np.prod(reg.shape))
        if reg == wire_reg:
            if reg_size == 1:
                # either shape = () or shape = (1,), wire_symbols is a list of
                # size reg.bitsize, we only want one label for the register.
                symbol = wire_symbols[begin]
            elif reg.bitsize > 1:
                # If the bitsize > 1 AND the shape of the register is non
                # trivial then we only want to index into the shape, (not shape
                # * bitsize)
                symbol = np.array(wire_symbols[begin : begin + reg_size]).reshape(reg.shape)[idx]
            else:
                # bitsize = 1 and shape is non trivial, index into the array of wireshapes.
                symbol = np.array(wire_symbols[begin:finish]).reshape(reg.shape)[idx]
        begin = finish
    return _cirq_wire_symbol_to_qualtran_wire_symbol(symbol, wire_reg.side)


def _my_tensors_from_gate(
    gate: cirq.Gate,
    signature: Signature,
    *,
    incoming: Dict[str, 'ConnectionT'],
    outgoing: Dict[str, 'ConnectionT'],
) -> List['qtn.Tensor']:
    import quimb.tensor as qtn

    from qualtran.simulation.tensor._dense import _order_incoming_outgoing_indices
    from qualtran.simulation.tensor._tensor_data_manipulation import (
        tensor_data_from_unitary_and_signature,
    )

    if not cirq.has_unitary(gate):
        raise NotImplementedError(f"Tensors are only supported for unitary gates, not {gate}.")

    unitary = tensor_data_from_unitary_and_signature(cirq.unitary(gate), signature)
    inds = _order_incoming_outgoing_indices(signature, incoming=incoming, outgoing=outgoing)
    unitary = unitary.reshape((2,) * len(inds))
    return [qtn.Tensor(data=unitary, inds=inds, tags=[str(gate)])]


@frozen(eq=False)
class _QReg:
    """Used as a container for qubits that form a `Register` of a given bitsize.

    Each instance of `_QReg` would correspond to a `Soquet` in Bloqs and represents an opaque collection
    of qubits that together form a quantum register.
    """

    qubits: Tuple[cirq.Qid, ...] = field(
        converter=lambda v: (v,) if isinstance(v, cirq.Qid) else tuple(v)
    )
    dtype: QDType

    # Overwrite hash / equality to ensure single-qubit registers map to each other correctly.
    # E.g., when updating qreg_to_qvars we may have output qregs with dtype =
    # QBit, but the input registers (which are used to track the qubits) may
    # have QUInt(1) or similar, leading to the mappings not updating correctly.
    # Single qubit QFxp cases are handled separately in _ensure_in_reg_exists
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _QReg):
            return False

        return other.qubits == self.qubits

    def __hash__(self):
        return hash(self.qubits)


def _ensure_in_reg_exists(
    bb: BloqBuilder, in_reg: _QReg, qreg_to_qvar: Dict[_QReg, Soquet]
) -> None:
    """Takes care of qubit allocations, split and joins to ensure `qreg_to_qvar[in_reg]` exists."""
    from qualtran.bloqs.bookkeeping import Cast

    all_mapped_qubits = {q for qreg in qreg_to_qvar for q in qreg.qubits}
    qubits_to_allocate: List[cirq.Qid] = [q for q in in_reg.qubits if q not in all_mapped_qubits]
    if qubits_to_allocate:
        n_alloc = len(qubits_to_allocate)
        qreg_to_qvar[
            _QReg(qubits_to_allocate, dtype=QBit() if n_alloc == 1 else QAny(n_alloc))
        ] = bb.allocate(n_alloc)

    if in_reg in qreg_to_qvar:
        # This is the easy case when no split / joins are needed.
        return

    # a. Split all registers containing at-least one qubit corresponding to `in_reg`.
    in_reg_qubits = set(in_reg.qubits)

    new_qreg_to_qvar: Dict[_QReg, Soquet] = {}
    for qreg, soq in qreg_to_qvar.items():
        if len(qreg.qubits) > 1 and any(q in qreg.qubits for q in in_reg_qubits):
            new_qreg_to_qvar |= {
                _QReg(q, QBit()): s for q, s in zip(qreg.qubits, bb.split(soq=soq))
            }
        else:
            new_qreg_to_qvar[qreg] = soq
    qreg_to_qvar.clear()

    # b. Join all 1-bit registers, corresponding to individual qubits, that make up `in_reg`.
    soqs_to_join: Dict[cirq.Qid, Soquet] = {}
    for qreg, soq in new_qreg_to_qvar.items():
        if len(in_reg_qubits) > 1 and qreg.qubits and qreg.qubits[0] in in_reg_qubits:
            assert len(qreg.qubits) == 1, "Individual qubits should have been split by now."
            # Cast single bit registers to QBit to preserve signature of later join.
            if not isinstance(qreg.dtype, QBit):
                soqs_to_join[qreg.qubits[0]] = bb.add(Cast(qreg.dtype, QBit()), reg=soq)
            else:
                soqs_to_join[qreg.qubits[0]] = soq
        elif len(in_reg_qubits) == 1 and qreg.qubits and qreg.qubits[0] in in_reg_qubits:
            # Cast single QBit registers to the appropriate single-bit register dtype.
            err_msg = (
                "Found non-QBit type register which shouldn't happen: "
                f"{soq.reg.name} {soq.reg.dtype}"
            )
            assert isinstance(soq.reg.dtype, QBit), err_msg
            if not isinstance(in_reg.dtype, QBit):
                qreg_to_qvar[in_reg] = bb.add(Cast(QBit(), in_reg.dtype), reg=soq)
            else:
                qreg_to_qvar[qreg] = soq
        else:
            qreg_to_qvar[qreg] = soq
    if soqs_to_join:
        # A split is not necessarily matched with a join of the same size so we
        # need to strip the data type of the parent split before assigning the correct bitsize.
        qreg_to_qvar[in_reg] = bb.join(
            np.array([soqs_to_join[q] for q in in_reg.qubits]), dtype=in_reg.dtype
        )


def _gather_input_soqs(
    bb: BloqBuilder, op_quregs: Dict[str, NDArray[_QReg]], qreg_to_qvar: Dict[_QReg, Soquet]  # type: ignore[type-var]
) -> Dict[str, NDArray[Soquet]]:  # type: ignore[type-var]
    qvars_in: Dict[str, NDArray[Soquet]] = {}  # type: ignore[type-var]
    for reg_name, quregs in op_quregs.items():
        flat_soqs: List[Soquet] = []
        for qureg in quregs.flatten():
            _ensure_in_reg_exists(bb, qureg, qreg_to_qvar)
            flat_soqs.append(qreg_to_qvar[qureg])
        qvars_in[reg_name] = np.array(flat_soqs).reshape(quregs.shape)
    return qvars_in


def cirq_gate_to_bloq(gate: cirq.Gate) -> Bloq:
    """For a given Cirq gate, return an equivalent bloq.

    This will try to find the idiomatically correct bloq to return. If there is no equivalent
    Qualtran bloq for the given Cirq gate, we wrap it in the `CirqGateAsBloq` wrapper class.
    """
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import (
        CHadamard,
        CNOT,
        CSwap,
        CYGate,
        CZ,
        CZPowGate,
        GlobalPhase,
        Hadamard,
        Identity,
        Rx,
        Ry,
        Rz,
        SGate,
        TGate,
        Toffoli,
        TwoBitSwap,
        XGate,
        XPowGate,
        YGate,
        YPowGate,
        ZGate,
        ZPowGate,
    )
    from qualtran.cirq_interop import CirqGateAsBloq
    from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate

    if isinstance(gate, BloqAsCirqGate):
        # Perhaps this operation was constructed from `Bloq.on()`.
        return gate.bloq
    if isinstance(gate, Bloq):
        # I.e., `GateWithRegisters`.
        return gate

    if isinstance(gate, cirq.ops.raw_types._InverseCompositeGate):
        # Inverse of a cirq gate, delegate to Adjoint
        return Adjoint(cirq_gate_to_bloq(gate._original))

    # Check specific basic gates instances.
    CIRQ_GATE_TO_BLOQ_MAP = {
        cirq.T: TGate(),
        cirq.T**-1: TGate().adjoint(),
        cirq.S: SGate(),
        cirq.S**-1: SGate().adjoint(),
        cirq.H: Hadamard(),
        cirq.ControlledGate(cirq.H): CHadamard(),
        cirq.CNOT: CNOT(),
        cirq.TOFFOLI: Toffoli(),
        cirq.X: XGate(),
        cirq.Y: YGate(),
        cirq.ControlledGate(cirq.Y): CYGate(),
        cirq.Z: ZGate(),
        cirq.CZ: CZ(),
        cirq.SWAP: TwoBitSwap(),
        cirq.CSWAP: CSwap(1),
        cirq.I: Identity(),
    }
    if gate in CIRQ_GATE_TO_BLOQ_MAP:
        return CIRQ_GATE_TO_BLOQ_MAP[gate]

    if isinstance(gate, cirq.ControlledGate):
        return cirq_gate_to_bloq(gate.sub_gate).controlled(
            ctrl_spec=CtrlSpec.from_cirq_cv(gate.control_values)
        )

    # Check specific basic gates types.
    CIRQ_TYPE_TO_BLOQ_MAP = {
        cirq.Rz: Rz,
        cirq.Rx: Rx,
        cirq.Ry: Ry,
        cirq.XPowGate: XPowGate,
        cirq.YPowGate: YPowGate,
        cirq.ZPowGate: ZPowGate,
        cirq.CZPowGate: CZPowGate,
    }
    if isinstance(gate, (cirq.Rx, cirq.Ry, cirq.Rz)):
        return CIRQ_TYPE_TO_BLOQ_MAP[gate.__class__](angle=gate._rads)

    if isinstance(gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.CZPowGate)):
        return CIRQ_TYPE_TO_BLOQ_MAP[gate.__class__](
            exponent=gate.exponent, global_shift=gate.global_shift
        )

    if isinstance(gate, cirq.GlobalPhaseGate):
        if isinstance(gate.coefficient, numbers.Complex):
            return GlobalPhase.from_coefficient(coefficient=complex(gate.coefficient))
        return GlobalPhase.from_coefficient(coefficient=gate.coefficient)

    # No known basic gate, wrap the cirq gate in a CirqGateAsBloq wrapper.
    return CirqGateAsBloq(gate)


def _extract_bloq_from_op(op: 'cirq.Operation') -> Bloq:
    """Get a `Bloq` out of a cirq Operation.

    Unwrap BloqAsCirqGate, pass through any GateWithRegisters, and wrap
    true cirq gates with `CirqGateAsBloq`.
    """
    if op.gate is None:
        raise ValueError(f"Only gate operations are supported, not {op}.")
    return cirq_gate_to_bloq(op.gate)


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

        - `Register('x', QAny(bitsize=2), shape=(3, 4), side=Side.THRU)`
        - `Register('y', QBit(), shape=(10, 20), side=Side.LEFT)`
        - `Register('z', QBit(), shape=(10, 20), side=Side.RIGHT)`

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
        signature = Signature([Register('qubits', QBit(), shape=(len(all_qubits),))])
        in_quregs = out_quregs = {'qubits': np.array(all_qubits).reshape(len(all_qubits), 1)}
    elif in_quregs is None or out_quregs is None:
        raise ValueError("`signature` requires specifying both `in_quregs` and `out_quregs`.")

    in_quregs: Dict[str, NDArray] = {
        k: np.apply_along_axis(_QReg, -1, *(v, signature.get_left(k).dtype))  # type: ignore[arg-type]
        for k, v in in_quregs.items()
    }
    out_quregs: Dict[str, NDArray] = {
        k: np.apply_along_axis(_QReg, -1, *(v, signature.get_right(k).dtype))  # type: ignore[arg-type]
        for k, v in out_quregs.items()
    }

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
        bloq = _extract_bloq_from_op(op)
        if bloq.signature == Signature([]):
            bb.add(bloq)
            continue

        reg_dtypes = [r.dtype for r in bloq.signature]
        # 3.1 Find input / output registers.
        all_op_quregs: Dict[str, NDArray[_QReg]] = {
            k: np.apply_along_axis(_QReg, -1, *(v, reg_dtypes[i]))  # type: ignore[arg-type]
            for i, (k, v) in enumerate(split_qubits(bloq.signature, op.qubits).items())
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


def decompose_from_cirq_style_method(
    bloq: Bloq, method_name: str = 'decompose_from_registers'
) -> CompositeBloq:
    """Return a `CompositeBloq` decomposition using a cirq-style decompose method.

    The bloq must have a method with the given name (by default: "decompose_from_registers") that
    satisfies the following function signature:

        def decompose_from_registers(
            self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
        ) -> cirq.OP_TREE:

    This must yield a list of `cirq.Operation`s using `cirq.Gate.on(...)`, `Bloq.on(...)`,
    `GateWithRegisters.on_registers(...)`, or `Bloq.on_registers(...)`.

    If `Bloq.on()` is used, the bloqs will be retained in their native form in the returned
    composite bloq. If `cirq.Gate.on()` is used, the gates will be wrapped in `CirqGateAsBloq`.

    Args:
        bloq: The bloq to decompose.
        method_name: The string name of the method that can be found on the bloq that
            yields the cirq-style decomposition.
    """
    if any(
        cirq.is_parameterized(reg.bitsize) or cirq.is_parameterized(reg.side) or reg.is_symbolic()
        for reg in bloq.signature
    ):
        # pylint: disable=raise-missing-from
        raise DecomposeTypeError(f"Cannot decompose parameterized {bloq}.")

    qm = InteropQubitManager()
    in_quregs = get_named_qubits(bloq.signature.lefts())
    qm.manage_qubits(itertools.chain.from_iterable(qreg.flatten() for qreg in in_quregs.values()))
    all_quregs, out_quregs = _get_all_and_output_quregs_from_input(bloq.signature, qm, in_quregs)
    context = cirq.DecompositionContext(qubit_manager=qm)
    dfr_method = getattr(bloq, method_name)
    decomposed_optree = dfr_method(context=context, **all_quregs)
    try:
        return cirq_optree_to_cbloq(
            decomposed_optree, signature=bloq.signature, in_quregs=in_quregs, out_quregs=out_quregs
        )
    except ValueError as exc:
        if "Only gate operations are supported" in str(exc):
            raise DecomposeNotImplementedError(str(exc)) from exc
        else:
            raise exc
