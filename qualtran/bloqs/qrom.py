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

from typing import Callable, Sequence, Set, Tuple

import attrs
import cirq
import numpy as np
from cirq._compat import cached_property
from numpy.typing import ArrayLike, NDArray

from qualtran import Register, SelectionRegister, Soquet
from qualtran._infra.gate_with_registers import merge_qubits, total_bits
from qualtran.bloqs.and_bloq import And, MultiAnd
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountT


@cirq.value_equality()
@attrs.frozen
class QROM(UnaryIterationGate):
    """Bloq to load data[l] in the target register when the selection stores an index l.

    In the case of multi-dimensional data[p,q,r,...] we use multiple named
    selection signature [p, q, r, ...] to index and load the data. Here `p, q, r, ...`
    correspond to signature named `selection0`, `selection1`, `selection2`, ... etc.

    When the input data elements contain consecutive entries of identical data elements to
    load, the QROM also implements the "variable-spaced" QROM optimization described in Ref[2].

    Args:
        data: List of numpy ndarrays specifying the data to load. If the length
            of this list is greater than one then we use the same selection indices
            to load each dataset (for example, to load alt and keep data for
            state preparation). Each data set is required to have the same
            shape and to be of integer type.
        selection_bitsizes: The number of bits used to represent each selection register
            corresponding to the size of each dimension of the array. Should be
            the same length as the shape of each of the datasets.
        target_bitsizes: The number of bits used to represent the data
            signature. This can be deduced from the maximum element of each of the
            datasets. Should be of length len(data), i.e. the number of datasets.
        num_controls: The number of control signature.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.

        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization]
        (https://arxiv.org/abs/2007.07391).
            Babbush et. al. (2020). Figure 3.
    """

    data: Sequence[NDArray]
    selection_bitsizes: Tuple[int, ...]
    target_bitsizes: Tuple[int, ...]
    num_controls: int = 0

    @classmethod
    def build(cls, *data: ArrayLike, num_controls: int = 0) -> 'QROM':
        _data = [np.array(d, dtype=int) for d in data]
        selection_bitsizes = tuple((s - 1).bit_length() for s in _data[0].shape)
        target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        return QROM(
            data=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    def __attrs_post_init__(self):
        shapes = [d.shape for d in self.data]
        assert all([isinstance(s, int) for s in self.selection_bitsizes])
        assert all([isinstance(t, int) for t in self.target_bitsizes])
        assert len(set(shapes)) == 1, f"Data must all have the same size: {shapes}"
        assert len(self.target_bitsizes) == len(self.data), (
            f"len(self.target_bitsizes)={len(self.target_bitsizes)} should be same as "
            f"len(self.data)={len(self.data)}"
        )
        assert all(
            t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data)
        )
        assert isinstance(self.selection_bitsizes, tuple)
        assert isinstance(self.target_bitsizes, tuple)

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if not self.num_controls else (Register('control', self.num_controls),)

    @cached_property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        ret = tuple(
            SelectionRegister(f'selection{i}', sb, l)
            for i, (l, sb) in enumerate(zip(self.data[0].shape, self.selection_bitsizes))
            if sb > 0
        )
        if len(self.data[0].shape) == 1 and len(ret) == 1:
            ret = (SelectionRegister('selection', ret[0].bitsize, ret[0].iteration_length),)
        return ret

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(Register(f'target{i}_', l) for i, l in enumerate(self.target_bitsizes) if l)

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: NDArray[cirq.Qid],  # type: ignore[type-var]
    ) -> cirq.OP_TREE:
        for i, d in enumerate(self.data):
            target = target_regs.get(f'target{i}_', ())
            for q, bit in zip(target, f'{int(d[selection_idx]):0{len(target)}b}'):
                if int(bit):
                    yield gate(q)

    def decompose_zero_selection(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> cirq.OP_TREE:
        controls = merge_qubits(self.control_registers, **quregs)
        target_regs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        zero_indx = (0,) * len(self.data[0].shape)
        if self.num_controls == 0:
            yield self._load_nth_data(zero_indx, cirq.X, **target_regs)
        elif self.num_controls == 1:
            yield self._load_nth_data(zero_indx, lambda q: cirq.CNOT(controls[0], q), **target_regs)
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
            yield self._load_nth_data(zero_indx, lambda q: cirq.CNOT(and_target, q), **target_regs)
            yield cirq.inverse(multi_controlled_and)
            context.qubit_manager.qfree(list(junk.flatten()) + [and_target])

    def _break_early(self, selection_index_prefix: Tuple[int, ...], l: int, r: int):
        for data in self.data:
            data_l_r_flat = data[selection_index_prefix][l:r].flat
            unique_element = data_l_r_flat[0]
            for x in data_l_r_flat:
                if x != unique_element:
                    return False
        return True

    def nth_operation(
        self, context: cirq.DecompositionContext, control: cirq.Qid, **kwargs
    ) -> cirq.OP_TREE:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        target_regs = {reg.name: kwargs[reg.name] for reg in self.target_registers}
        yield self._load_nth_data(selection_idx, lambda q: cirq.CNOT(control, q), **target_regs)

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["@"] * self.num_controls
        wire_symbols += ["In"] * total_bits(self.selection_registers)
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROM_{i}"] * target.total_bits()
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        name = soq.reg.name
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
            return TextBox(f'data_{subscript}')
        elif name == 'control':
            return Circle()

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def _value_equality_values_(self):
        data_tuple = tuple(tuple(d.flatten()) for d in self.data)
        return (self.selection_registers, self.target_registers, self.control_registers, data_tuple)

    def nth_operation_callgraph(self, **kwargs: int) -> Set['BloqCountT']:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        return {(CNOT(), sum(int(d[selection_idx]).bit_count() for d in self.data))}
