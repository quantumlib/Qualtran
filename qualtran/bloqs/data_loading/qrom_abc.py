#  Copyright 2024 Google LLC
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

"""Base class for Bloqs implementing a QROM (Quantum read-only memory) circuit."""
import abc
from functools import cached_property
from typing import Dict, Tuple, Type, TypeVar, Union

import attrs
import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qualtran import BoundedQUInt, QAny, Register
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import bit_length, is_symbolic, shape, Shaped, SymbolicInt

QROMT = TypeVar('QROMT', bound='QROMABC')


@cirq.value_equality(distinct_child_types=True)
@attrs.frozen
class QROMABC(abc.ABC):
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

    data_or_shape: Tuple[Union[NDArray, Shaped], ...] = attrs.field(
        converter=lambda x: tuple(np.array(y) if isinstance(y, (list, tuple)) else y for y in x)
    )
    selection_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    target_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    target_shapes: Tuple[Union[Shaped, Tuple[SymbolicInt, ...]], ...] = attrs.field(
        converter=lambda x: tuple(tuple(y) for y in x)
    )
    num_controls: SymbolicInt = 0

    @target_shapes.default
    def _default_target_shapes(self):
        return ((),) * len(self.data_or_shape)

    @cached_property
    def data_shape(self) -> Tuple[SymbolicInt, ...]:
        ret = None
        for data_or_shape, target_shape in zip(self.data_or_shape, self.target_shapes):
            data_shape = shape(data_or_shape)
            if target_shape:
                data_shape = data_shape[: -len(target_shape)]
            if ret is None:
                ret = data_shape
            if tuple(ret) != tuple(data_shape):
                raise ValueError("All datasets must have same shape")
        return ret

    def has_data(self) -> bool:
        return all(isinstance(d, np.ndarray) for d in self.data_or_shape)

    @property
    def data(self) -> Tuple[np.ndarray, ...]:
        if not self.has_data():
            raise ValueError(f"Data not available for symbolic QROM {self}")
        assert all(isinstance(d, np.ndarray) for d in self.data_or_shape)
        return self.data_or_shape

    def __attrs_post_init__(self):
        assert all([is_symbolic(s) or isinstance(s, int) for s in self.selection_bitsizes])
        assert all([is_symbolic(t) or isinstance(t, int) for t in self.target_bitsizes])
        assert len(self.target_bitsizes) == len(self.data_or_shape), (
            f"len(self.target_bitsizes)={len(self.target_bitsizes)} should be same as "
            f"len(self.data)={len(self.data_or_shape)}"
        )
        assert len(self.target_bitsizes) == len(self.target_shapes)
        if isinstance(self.data_or_shape, np.ndarray) and not is_symbolic(*self.target_bitsizes):
            assert all(
                t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, self.data)
            )
        assert isinstance(self.selection_bitsizes, tuple)
        assert isinstance(self.target_bitsizes, tuple)

    @classmethod
    def _build_from_data(
        cls: Type[QROMT],
        *data: ArrayLike,
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]] = None,
        num_controls: SymbolicInt = 0,
    ) -> QROMT:
        _data = [np.array(d, dtype=int) for d in data]
        selection_bitsizes = tuple((s - 1).bit_length() for s in _data[0].shape)
        if target_bitsizes is None:
            target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        return cls(
            data_or_shape=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            num_controls=num_controls,
        )

    def with_data(self, *data: ArrayLike) -> 'QROM':
        _data = tuple([np.array(d, dtype=int) for d in data])
        assert all(shape(d1) == shape(d2) for d1, d2 in zip(_data, self.data_or_shape))
        assert len(_data) == len(self.target_bitsizes)
        assert all(t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, _data))
        return attrs.evolve(self, data_or_shape=_data)

    @classmethod
    def _build_from_bitsize(
        cls: Type[QROMT],
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        target_shapes: Tuple[Union[Shaped, Tuple[SymbolicInt, ...]], ...] = (),
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QROMT:
        data_shape = (
            (data_len_or_shape,) if isinstance(data_len_or_shape, int) else data_len_or_shape
        )
        if not isinstance(target_bitsizes, tuple):
            target_bitsizes = (target_bitsizes,)
        if target_shapes == ():
            target_shapes = ((),) * len(target_bitsizes)
        _data = [Shaped(data_shape + sh) for sh in target_shapes]
        if selection_bitsizes == ():
            selection_bitsizes = tuple(bit_length(s - 1) for s in data_shape)
        assert len(selection_bitsizes) == len(data_shape)
        return cls(
            data_or_shape=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            target_shapes=target_shapes,
            num_controls=num_controls,
        )

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return () if not self.num_controls else (Register('control', QAny(self.num_controls)),)

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        types = [
            BoundedQUInt(sb, l)
            for l, sb in zip(self.data_shape, self.selection_bitsizes)
            if is_symbolic(sb) or sb > 0
        ]
        if len(types) == 1:
            return (Register('selection', types[0]),)
        return tuple(Register(f'selection{i}', qdtype) for i, qdtype in enumerate(types))

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Register(f'target{i}_', QAny(l), shape=sh)
            for i, (l, sh) in enumerate(zip(self.target_bitsizes, self.target_shapes))
            if is_symbolic(l) or l
        )

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

    def _value_equality_values_(self):
        data_tuple = (
            tuple(tuple(d.flatten()) for d in self.data) if self.has_data() else self.data_or_shape
        )
        return (self.selection_registers, self.target_registers, self.control_registers, data_tuple)

    def __pow__(self, power: int):
        if power in [1, -1]:
            return self
        return NotImplemented  # pragma: no cover
