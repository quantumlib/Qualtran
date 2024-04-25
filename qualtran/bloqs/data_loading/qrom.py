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
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import attrs
import cirq
import numpy as np
import sympy
from numpy.typing import ArrayLike, NDArray

from qualtran import bloq_example, BloqDocSpec, BoundedQUInt, QAny, Register
from qualtran._infra.gate_with_registers import merge_qubits
from qualtran.bloqs.basic_gates import CNOT
from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
from qualtran.bloqs.multiplexers.unary_iteration_bloq import UnaryIterationGate
from qualtran.drawing import Circle, Text, TextBox, WireSymbol
from qualtran.resource_counting import BloqCountT
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import bit_length, is_symbolic, prod, shape, Shaped, SymbolicInt

if TYPE_CHECKING:
    from qualtran.resource_counting import SympySymbolAllocator


def _to_tuple(x: Iterable[NDArray]) -> Sequence[NDArray]:
    """Needed so mypy can correctly infer types."""
    return tuple(x)


@cirq.value_equality()
@attrs.frozen
class QROM(UnaryIterationGate):
    r"""Bloq to load `data[l]` in the target register when the selection stores an index `l`.

    ## Overview
    The action of a QROM can be described as
    $$
            \text{QROM}_{s_1, s_2, \dots, s_K}^{d_1, d_2, \dots, d_L}
            |s_1\rangle |s_2\rangle \dots |s_K\rangle
            |0\rangle^{\otimes b_1} |0\rangle^{\otimes b_2} \dots |0\rangle^{\otimes b_L}
            \rightarrow
            |s_1\rangle |s_2\rangle \dots |s_K\rangle
            |d_1[s_1, s_2, \dots, s_k]\rangle
            |d_2[s_1, s_2, \dots, s_k]\rangle \dots
            |d_L[s_1, s_2, \dots, s_k]\rangle
    $$

    There two high level parameters that control the behavior of a QROM are -

    1. Shape of the classical dataset to be loaded ($\text{data.shape} = (S_1, S_2, ..., S_K)$).
    2. Number of distinct datasets to be loaded ($\text{data.bitsizes} = (b_1, b_2, ..., b_L)$).

    Each of these have an effect on the cost of the QROM. The `data_or_shape` parameter stores
    either
    1. A numpy array of shape $(L, S_1, S_2, ..., S_K)$ when $L$ classical datasets, each of
       shape $(S_1, S_2, ..., S_K)$ and bitsizes $(b_1, b_2, ..., b_L)$ are to be loaded and
       the classical data is available to instantiate the QROM bloq. In this case, the helper
       builder `QROM.build_from_data(data_1, data_2, ..., data_L)` can be used to build the QROM.

    2. A `Shaped` object that stores a (potentially symbolic) tuple $(L, S_1, S_2, ..., S_K)$
       that represents the number of classical datasets `L=data_or_shape.shape[0]` and
       their shape `data_shape=data_or_shape.shape[1:]` to be loaded by this QROM. This is used
       to instantiate QROM bloqs for symbolic cost analysis where the exact data to be loaded
       is not known. In this case, the helper builder `QROM.build_from_bitsize` can be used
       to build the QROM.

    ### Shape of the classical dataset to be loaded.
    QROM bloq supports loading multidimensional classical datasets. In order to load a data
    set of shape $\mathrm{data.shape} == (P, Q, R, S)$ the QROM bloq needs four selection
    registers with bitsizes $(p, q, r, s)$ where
    $p,q,r,s=\log_2{P}, \log_2{Q}, \log_2{R}, \log_2{S}$.

    In general, to load K dimensional data, we use K named selection registers `(selection0,
    selection1, ..., selection{k})` to index and load the data.

    The T/Toffoli cost of the QROM scales linearly with the number of elements in the dataset
    (i.e. $\mathcal{O}(\mathrm{np.prod(data.shape)}$).

    ### Number of distinct datasets to be loaded, and their corresponding target bitsize.
    To load a classical dataset into a target register of bitsize $b$, the clifford cost of a QROM
    scales as $\mathcal{O}(b \mathrm{np.prod}(\mathrm{data.shape}))$. This is because we need
    $\mathcal{O}(b)$ CNOT gates to load the ith data element in the target register when the
    selection register stores index $i$.

    If you have multiple classical datasets `(data_1, data_2, data_3, ..., data_L)` to be loaded
    and each of them has the same shape `(data_1.shape == data_2.shape == ... == data_L.shape)`
    and different target bitsizes `(b_1, b_2, ..., b_L)`, then one construct a single classical
    dataset `data = merge(data_1, data_2, ..., data_L)` where

    - `data.shape == data_1.shape == data_2.shape == ... == data_L` and
    - `data[idx] = f'{data_1[idx]!0{b_1}b}' + f'{data_2[idx]!0{b_2}b}' + ... + f'{data_L[idx]!0{b_L}b}'`

    Thus, the target bitsize of the merged dataset is $b = b_1 + b_2 + \dots + b_L$ and clifford
    cost of loading merged dataset scales as
    $\mathcal{O}((b_1 + b_2 + \dots + b_L) \mathrm{np.prod}(\mathrm{data.shape}))$.

    ## Variable spaced QROM
    When the input classical data contains consecutive entries of identical data elements to
    load, the QROM also implements the "variable-spaced" QROM optimization described in Ref [2].

    Args:
        data_or_shape: List of numpy ndarrays specifying the data to load. If the length
            of this list ($L$) is greater than one then we use the same selection indices
            to load each dataset. Each data set is required to have the same
            shape $(S_1, S_2, ..., S_K)$ and to be of integer type. For symbolic QROMs,
            pass a `Shaped` object instead with shape $(L, S_1, S_2, ..., S_K)$.
        selection_bitsizes: The number of bits used to represent each selection register
            corresponding to the size of each dimension of the array $(S_1, S_2, ..., S_K)$.
            Should be the same length as the shape of each of the datasets.
        target_bitsizes: The number of bits used to represent the data signature. This can be
            deduced from the maximum element of each of the datasets. Should be a tuple
            $(b_1, b_2, ..., b_L)$ of length `L = len(data)`, i.e. the number of datasets to
            be loaded.
        num_controls: The number of controls.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.

        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
            Babbush et. al. (2020). Figure 3.
    """

    data_or_shape: Union[NDArray, Shaped] = attrs.field(
        converter=lambda x: np.array(x) if isinstance(x, (list, tuple)) else x
    )
    selection_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    target_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    num_controls: SymbolicInt = 0

    def has_data(self) -> bool:
        return not isinstance(self.data_or_shape, Shaped)

    @property
    def data_shape(self) -> Tuple[SymbolicInt, ...]:
        return shape(self.data_or_shape)[1:]

    @property
    def data(self) -> np.ndarray:
        if not self.has_data():
            raise ValueError(f"Data not available for symbolic QROM {self}")
        assert isinstance(self.data_or_shape, np.ndarray)
        return self.data_or_shape

    def __attrs_post_init__(self):
        assert all([is_symbolic(s) or isinstance(s, int) for s in self.selection_bitsizes])
        assert all([is_symbolic(t) or isinstance(t, int) for t in self.target_bitsizes])
        assert len(self.target_bitsizes) == self.data_or_shape.shape[0], (
            f"len(self.target_bitsizes)={len(self.target_bitsizes)} should be same as "
            f"len(self.data)={self.data_or_shape.shape[0]}"
        )
        if isinstance(self.data_or_shape, np.ndarray) and not is_symbolic(*self.target_bitsizes):
            assert all(
                t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data)
            )
        assert isinstance(self.selection_bitsizes, tuple)
        assert isinstance(self.target_bitsizes, tuple)

    @classmethod
    def build_from_data(cls, *data: ArrayLike, num_controls: SymbolicInt = 0) -> 'QROM':
        _data = np.array([np.array(d, dtype=int) for d in data])
        selection_bitsizes = tuple((s - 1).bit_length() for s in _data.shape[1:])
        target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        return QROM(
            data_or_shape=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    @classmethod
    def build_from_bitsize(
        cls,
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> 'QROM':
        data_shape = (
            (data_len_or_shape,) if isinstance(data_len_or_shape, int) else data_len_or_shape
        )
        if not isinstance(target_bitsizes, tuple):
            target_bitsizes = (target_bitsizes,)
        _data = Shaped((len(target_bitsizes),) + data_shape)
        if selection_bitsizes is ():
            selection_bitsizes = tuple(bit_length(s - 1) for s in _data.shape[1:])
        assert len(selection_bitsizes) == len(_data.shape) - 1
        return QROM(
            data_or_shape=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if not self.num_controls else (Register('control', QAny(self.num_controls)),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        types = [
            BoundedQUInt(sb, l)
            for l, sb in zip(self.data_or_shape.shape[1:], self.selection_bitsizes)
            if is_symbolic(sb) or sb > 0
        ]
        if len(types) == 1:
            return (Register('selection', types[0]),)
        return tuple(Register(f'selection{i}', qdtype) for i, qdtype in enumerate(types))

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Register(f'target{i}_', QAny(l))
            for i, l in enumerate(self.target_bitsizes)
            if is_symbolic(l) or l
        )

    def _load_nth_data(
        self,
        selection_idx: Tuple[int, ...],
        gate: Callable[[cirq.Qid], cirq.Operation],
        **target_regs: NDArray[cirq.Qid],  # type: ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        for i, d in enumerate(self.data):
            target = target_regs.get(f'target{i}_', ())
            for q, bit in zip(target, f'{int(d[selection_idx]):0{len(target)}b}'):
                if int(bit):
                    yield gate(q)

    def decompose_zero_selection(
        self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]
    ) -> Iterator[cirq.OP_TREE]:
        controls = merge_qubits(self.control_registers, **quregs)
        target_regs = {reg.name: quregs[reg.name] for reg in self.target_registers}
        zero_indx = (0,) * len(self.data_shape)
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
        yield self._load_nth_data(selection_idx, lambda q: CNOT().on(control, q), **target_regs)

    def on_classical_vals(self, **vals: 'ClassicalValT') -> Dict[str, 'ClassicalValT']:
        if not self.has_data():
            raise NotImplementedError(f'Symbolic {self} does not support classical simulation')

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
            idx = tuple(vals[f'selection{i}'] for i in range(n_dim))  # type: ignore[assignment]
            selections = {f'selection{i}': idx[i] for i in range(n_dim)}  # type: ignore[index]

        # Retrieve the data; bitwise add them in to the input target values
        targets = {f'target{d_i}_': d[idx] for d_i, d in enumerate(self.data)}
        targets = {k: v ^ vals[k] for k, v in targets.items()}
        return controls | selections | targets

    def my_static_costs(self, cost_key: 'CostKey') -> Union[Any, NotImplemented]:
        if cost_key == QubitCount():
            return self.num_controls + 2 * sum(self.selection_bitsizes) + sum(self.target_bitsizes)
        return super().my_static_costs(cost_key)

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

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover

    def _value_equality_values_(self):
        data_tuple = (
            tuple(tuple(d.flatten()) for d in self.data) if self.has_data() else self.data_or_shape
        )
        return (self.selection_registers, self.target_registers, self.control_registers, data_tuple)

    def nth_operation_callgraph(self, **kwargs: int) -> Set['BloqCountT']:
        selection_idx = tuple(kwargs[reg.name] for reg in self.selection_registers)
        return {(CNOT(), sum(int(d[selection_idx]).bit_count() for d in self.data))}

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> Set['BloqCountT']:
        if self.has_data():
            return super().build_call_graph(ssa=ssa)
        n_and = prod(*self.data_shape) - 2 + self.num_controls
        n_cnot = prod(*self.target_bitsizes, *self.data_shape)
        return {(And(), n_and), (And().adjoint(), n_and), (CNOT(), n_cnot)}

    def __str__(self):
        return 'QROM'


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
