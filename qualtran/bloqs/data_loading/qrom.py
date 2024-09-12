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

"""Quantum read-only memory."""
import numbers
from typing import cast, Iterable, Iterator, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import ArrayLike, NDArray

from qualtran import bloq_example, BloqDocSpec, DecomposeTypeError, QUInt, Register
from qualtran._infra.gate_with_registers import merge_qubits
from qualtran.bloqs.arithmetic import XorK
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.data_loading.qrom_base import QROMBase
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.symbolics import prod, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import BloqCountDictT, BloqCountT, SympySymbolAllocator


def _to_tuple(x: Iterable[NDArray]) -> Sequence[NDArray]:
    """Needed so mypy can correctly infer types."""
    return tuple(x)


@attrs.frozen
class QROM(QROMBase, UnaryIterationGate):  # type: ignore[misc]
    r"""Bloq to load `data[l]` in the target register when the selection stores an index `l`.

    See docstrings of `QROMBase` for an overview of the QROM primitive and the various attributes.

    This bloq is an implementation of the `QROMBase` interface that uses the unary iteration based
    approach described in Ref [1].

    ## Cost of this (unary iteration based) QROM

    ### T / Toffoli cost
    The T/Toffoli cost of this QROM scales linearly with the product of iteration lengths over
    all dimensions (i.e. $\mathcal{O}(\mathrm{np.prod(\text{selection\_shape})}$).

    ### Clifford Cost
    To load a classical dataset into a target register of bitsize $b$ and shape
    $\text{target\_shape}$, the clifford cost of this QROM scales as
    $\mathcal{O}(b \cdot \text{np.prod(selection\_shape+target\_shape)})
    =\mathcal{O}(b \cdot \text{np.prod(data.shape)})$. This is because we need $\mathcal{O}(b)$
    CNOT gates to load 1 classical data element in the target register and for each of the
    $\text{np.prod(selection\_shape)}$ selection indices, we have $\text{np.prod(target\_shape)}$
    such data elements to load.

    ### Ancilla cost
    The number of clean ancilla required by this QROM scales linearly with the size of the
    selection registers + number of controls.

    ## Variable spaced QROM
    When the input classical data contains consecutive entries of identical data elements to
    load, the QROM also implements the "variable-spaced" QROM optimization described in Ref [2].

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.

        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
            Babbush et. al. (2020). Figure 3.
    """

    @classmethod
    def build_from_data(
        cls,
        *data: ArrayLike,
        target_bitsizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> 'QROM':
        return cls._build_from_data(
            *data,
            target_bitsizes=target_bitsizes,
            target_shapes=target_shapes,
            num_controls=num_controls,
        )

    @classmethod
    def build_from_bitsize(
        cls,
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> 'QROM':
        return cls._build_from_bitsize(
            data_len_or_shape,
            target_bitsizes,
            target_shapes=target_shapes,
            selection_bitsizes=selection_bitsizes,
            num_controls=num_controls,
        )

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        ctrl_qubits: Tuple[cirq.Qid, ...] = (),
        **target_regs: NDArray[cirq.Qid],  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        for i, d in enumerate(self.data):
            target = target_regs.get(f'target{i}_', np.array([]))
            target_bitsize, target_shape = self.target_bitsizes[i], self.target_shapes[i]
            assert all(isinstance(x, (int, numbers.Integral)) for x in target_shape)
            for idx in np.ndindex(cast(Tuple[int, ...], target_shape)):
                data_to_load = int(d[selection_idx + idx])
                yield XorK(QUInt(target_bitsize), data_to_load).on(*target[idx]).controlled_by(
                    *ctrl_qubits
                )

    def decompose_zero_selection(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        controls = tuple(merge_qubits(self.control_registers, **quregs))
        target_regs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        zero_indx = (0,) * len(self.data_shape)
        if self.num_controls <= 1:
            yield self._load_nth_data(zero_indx, ctrl_qubits=controls, **target_regs)
        else:
            ctrl = np.array(controls)[:, np.newaxis]
            junk = np.array(context.qubit_manager.qalloc(len(controls) - 2))[:, np.newaxis]
            and_target = context.qubit_manager.qalloc(1)[0]
            if self.num_controls == 2:
                multi_controlled_and = And().on_registers(ctrl=ctrl, target=and_target)
            else:
                multi_controlled_and = MultiAnd((1,) * len(controls)).on_registers(
                    ctrl=ctrl, junk=junk, target=and_target
                )
            yield multi_controlled_and
            yield self._load_nth_data(zero_indx, ctrl_qubits=(and_target,), **target_regs)
            yield cirq.inverse(multi_controlled_and)
            context.qubit_manager.qfree(list(junk.flatten()) + [and_target])

    def decompose_from_registers(
        self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        if self.has_data():
            return super().decompose_from_registers(context=context, **quregs)
        raise DecomposeTypeError(f"Cannot decompose symbolic {self} with no data.")

    def _break_early(self, selection_index_prefix: Tuple[int, ...], l: int, r: int):
        if not self.has_data():
            return False

        for data in self.data:
            data_l_r_flat = data[selection_index_prefix][l:r].flat
            unique_element = data_l_r_flat[0]
            for x in data_l_r_flat:
                if x != unique_element:
                    return False
        return True

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> Iterator[cirq.OP_TREE]:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = {reg.name: kwargs[reg.name] for reg in self.target_registers}
        yield self._load_nth_data(selection_idx, ctrl_qubits=(control,), **target_regs)

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        from qualtran.cirq_interop._bloq_to_cirq import _wire_symbol_to_cirq_diagram_info

        return _wire_symbol_to_cirq_diagram_info(self, args)

    def wire_symbol(self, reg: Optional[Register], idx: Tuple[int, ...] = tuple()) -> 'WireSymbol':
        if reg is None:
            return Text('QROM')
        name = reg.name
        if name == 'selection':
            return TextBox('In')
        elif 'selection' in name:
            sel_indx = int(name.replace('selection', ''))
            # get i,j,k,l,m,n type subscripts
            subscript = chr(ord('i') + sel_indx)
            return TextBox(f'In_{subscript}')
        elif 'target' in name:
            trg_indx = int(name.replace('target', '').replace('_', ''))
            # match the sel index
            subscript = chr(ord('a') + trg_indx)
            return TextBox(f'QROM_{subscript}')
        elif name == 'control':
            return Circle()
        raise ValueError(f'Unrecognized register name {name}')

    def nth_operation_callgraph(self, **kwargs: int) -> Set['BloqCountT']:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        ret = 0
        for i, d in enumerate(self.data):
            target_bitsize, target_shape = self.target_bitsizes[i], self.target_shapes[i]
            assert all(isinstance(x, (int, numbers.Integral)) for x in target_shape)
            for idx in np.ndindex(cast(Tuple[int, ...], target_shape)):
                data_to_load = int(d[selection_idx + idx])
                ret += data_to_load.bit_count()
        return {(CNOT(), ret)}

    def build_call_graph(
        self, ssa: 'SympySymbolAllocator'
    ) -> Union['BloqCountDictT', Set['BloqCountT']]:
        if self.has_data():
            return super().build_call_graph(ssa=ssa)
        n_and = prod(self.data_shape) - 2 + self.num_controls
        n_cnot = prod(
            bitsize * prod(sh) for bitsize, sh in zip(self.target_bitsizes, self.target_shapes)
        ) * prod(self.data_shape)
        return {And(): n_and, And().adjoint(): n_and, CNOT(): n_cnot}

    def adjoint(self) -> 'QROM':
        return self


@bloq_example
def _qrom_small() -> QROM:
    data = np.arange(5)
    qrom_small = QROM([data], selection_bitsizes=(3,), target_bitsizes=(3,))
    return qrom_small


@bloq_example
def _qrom_multi_data() -> QROM:
    data1 = np.arange(5)
    data2 = np.arange(5) + 1
    qrom_multi_data = QROM([data1, data2], selection_bitsizes=(3,), target_bitsizes=(3, 4))
    return qrom_multi_data


@bloq_example
def _qrom_multi_dim() -> QROM:
    data1 = np.arange(9).reshape((3, 3))
    data2 = (np.arange(9) + 1).reshape((3, 3))
    qrom_multi_dim = QROM([data1, data2], selection_bitsizes=(2, 2), target_bitsizes=(8, 8))
    return qrom_multi_dim


@bloq_example
def _qrom_symb() -> QROM:
    N, M, b1, b2, c = sympy.symbols('N M b1 b2 c')
    qrom_symb = QROM.build_from_bitsize((N, M), (b1, b2), num_controls=c)
    return qrom_symb


_QROM_DOC = BloqDocSpec(
    bloq_cls=QROM,
    import_line='from qualtran.bloqs.data_loading.qrom import QROM',
    examples=[_qrom_small, _qrom_multi_data, _qrom_multi_dim, _qrom_symb],
)
