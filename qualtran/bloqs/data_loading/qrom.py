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

from functools import cached_property
from typing import Callable, Dict, Sequence, Set, Tuple

import attrs
import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qualtran import bloq_example, BloqDocSpec, BoundedQUInt, QAny, Register, Soquet
from qualtran._infra.gate_with_registers import merge_qubits, total_bits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.unary_iteration_bloq import UnaryIterationGate
from qualtran.drawing import Circle, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountT
from qualtran.simulation.classical_sim import ClassicalValT


@cirq.value_equality()
@attrs.frozen
class QROM(UnaryIterationGate):
    """Bloq to load `data[l]` in the target register when the selection stores an index `l`.

    In the case of multidimensional `data[p,q,r,...]` we use multiple named
    selection registers to index and load the data named selection0, selection1, ...

    When the input data elements contain consecutive entries of identical data elements to
    load, the QROM also implements the "variable-spaced" QROM optimization described in Ref [2].

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
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.

        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
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
        return () if not self.num_controls else (Register('control', QAny(self.num_controls)),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        types = [
            BoundedQUInt(sb, l)
            for l, sb in zip(self.data[0].shape, self.selection_bitsizes)
            if sb > 0
        ]
        if len(types) == 1:
            return (Register('selection', types[0]),)
        return tuple(Register(f'selection{i}', qdtype) for i, qdtype in enumerate(types))

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Register(f'target{i}_', QAny(l)) for i, l in enumerate(self.target_bitsizes) if l
        )

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
            yield self._load_nth_data(zero_indx, lambda q: CNOT().on(controls[0], q), **target_regs)
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
            yield self._load_nth_data(zero_indx, lambda q: CNOT().on(and_target, q), **target_regs)
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
        yield self._load_nth_data(selection_idx, lambda q: CNOT().on(control, q), **target_regs)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if self.num_controls > 0:
            control = vals['control']
            if control != 2**self.num_controls - 1:
                return vals
            controls = {'control': control}
        else:
            controls = {}

        n_dim = len(self.selection_bitsizes)
        if n_dim == 1:
            idx = vals['selection']
            selections = {'selection': idx}
        else:
            # Multidimensional
            idx = tuple(vals[f'selection{i}'] for i in range(n_dim))
            selections = {f'selection{i}': idx[i] for i in range(n_dim)}

        # Retrieve the data; bitwise add them in to the input target values
        targets = {f'target{d_i}_': d[idx] for d_i, d in enumerate(self.data)}
        targets = {k: v ^ vals[k] for k, v in targets.items()}
        return controls | selections | targets

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


_QROM_DOC = BloqDocSpec(
    bloq_cls=QROM,
    import_line='from qualtran.bloqs.qrom import QROM',
    examples=[_qrom_small, _qrom_multi_data, _qrom_multi_dim],
)
