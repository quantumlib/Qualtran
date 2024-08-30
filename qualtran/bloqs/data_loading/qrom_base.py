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
import numbers
from functools import cached_property
from typing import cast, Dict, Optional, Tuple, Type, TypeVar, Union

import attrs
import numpy as np
import sympy
from numpy.typing import ArrayLike, NDArray

from qualtran import BloqDocSpec, BQUInt, QAny, Register, Side
from qualtran.simulation.classical_sim import ClassicalValT
from qualtran.symbolics import bit_length, is_symbolic, shape, Shaped, SymbolicInt

QROM_T = TypeVar('QROM_T', bound='QROMBase')


def _data_or_shape_to_tuple(data_or_shape: Tuple[Union[NDArray, Shaped], ...]) -> Tuple:
    return tuple(tuple(d.flatten()) if isinstance(d, np.ndarray) else d for d in data_or_shape)


@attrs.frozen
class QROMBase(metaclass=abc.ABCMeta):
    r"""Interface for Bloqs to load `data[l]` when the selection register stores index `l`.

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

    A behavior of a QROM can be understood in terms of its classical analogue, where a for-loop
    over one or more (selection) indices can be used to load one or more classical datasets, where
    each of the classical dataset can be multidimensional.

    ```
    >>> # N, M, P, Q, R, S, T are pre-initialized integer parameters.
    >>> output = [np.zeros((P, Q)), np.zeros((R, S, T))]
    >>> # Load two different classical datasets; each of different shape.
    >>> data = [np.random.rand(N, M, P, Q), np.random.rand(N, M, R, S, T)]
    >>> for i in range(N): # For loop over two selection indices i and j.
    >>>     for j in range(M):
    >>>        # Load two multidimensional classical datasets data[0] and data[1] s.t.
    >>>        # |i, j⟩|0⟩  -> |i, j⟩|data[0][i, j, :]⟩|data[1][i, j, :]⟩
    >>>        output[0] = data[0][i, j, :]
    >>>        output[1] = data[1][i, j, :]
    ```

    The parameters that control the behavior and costs of a QROM are -

    1. Number of selection registers (eg: $i$, $j$) and their iteration lengths (eg: $N$, $M$).
    2. Number of target registers, their quantum datatype and shape.
        - Number of target registers: One for each classical dataset to load (eg: $\text{data}[0]$
            and $\text{data}[1]$)
        - QDType of target registers: Depends on `dtype` of the $i$'th classical dataset
        - Shape of target registers: Depends on shape of classical data (eg: $(P, Q)$ and
            $(R, S, T)$ above)

    ### Specification of classical data via `data_or_shape`
    Users can specify the classical data to load via QROM by passing in an appropriate value
    for `data_or_shape` attribute. This is a list of numpy arrays or `Shaped` objects, where
    each item of the list corresponds to a classical dataset to load.

    Each classical dataset to load can be specified as a numpy array (or a `Shaped` object for
    symbolic bloqs). The shape of the dataset is a union of the selection shape and target shape,
    s.t.
    $$
        \text{data[i].shape} = \text{selection\_shape} + \text{target\_shape[i]}
    $$

    Note that the $\text{selection\_shape}$ should be same across all classical datasets to be
    loaded and correspond to a tuple of iteration lengths of selection indices (i.e. $(N, M)$
    in the example above).

    The target shape of each classical dataset can be different and parameterizes the size of
    the desired output that should be loaded in a target register.

    ### Number of selection registers and their iteration lengths
    As describe in the previous section, the number of selection registers and their iteration
    lengths can be inferred from the shape of the classical dataset. All classical datasets
    to be loaded must have the same $\text{selection\_shape}$, which is a tuple of iteration
    lengths over each dimension of the dataset (i.e. the range for each nested for-loop).

    In order to load a data set with $\text{selection\_shape} == (P, Q, R, S)$ the QROM bloq
    needs four selection registers with bitsizes $(p, q, r, s)$ where each of
    $p,q,r,s \geq \log_2{P}, \log_2{Q}, \log_2{R}, \log_2{S}$.

    In general, to load $K$ dimensional data, we use $K$ named selection registers
    $(\text{selection}_0, \text{selection}_1, ..., \text{selection}_k)$ to index and
    load the data. For the $i$'th selection register, its size is configured using
    attribute $\text{selection\_bitsizes[i]}$ and the iteration range is configued
    using $\text{data\_or\_shape[0].shape[i]}$.

    ### Number of target registers, their quantum datatype and shape
    QROM bloq uses one target register for each entry corresponding to classical dataset in the
    tuple `data_or_shape`. Thus, to load $L$ classical datsets, we use $L$ names target registers
    $(\text{target}_0, \text{target}_1, ..., \text{target}_L)$

    Each named target register has a bitsize $b_{i}=\text{target\_bitsizes[i]}$ that represents
    the size of the register and depends upon the maximum value of individual elements in the
    $i$'th classical dataset.

    Each named target register has a shape that can be configured using attribute
    $\text{target\_shape[i]}$ that represents the number of target registers if the output to load
    is multidimensional.

    Args:
        data_or_shape: List of numpy ndarrays specifying the data to load. If the length
            of this list ($L$) is greater than one then we use the same selection indices
            to load each dataset. The shape of a classical dataset is a concatenation of
            selection_shape and target_shape[i]; i.e. `data_or_shape[i].shape =
            selection_shape + target_shape[i]`. Thus, each data set is required to have the
            same selection shape $(S_1, S_2, ..., S_K)$ and can have a different
            target shape given by `target_shapes[i]`. For symbolic QROMs, pass a list of
            `Shaped` objects instead with shape $(S_1, S_2, ..., S_K) + target_shape[i]$.
        selection_bitsizes: The number of bits used to represent each selection register
            corresponding to the size of each dimension of the selection_shape
            $(S_1, S_2, ..., S_K)$. Should be the same length as the selection shape of
            each of the datasets and $2**\text{selection\_bitsizes[i]} >= S_i$
        target_shapes: Shape of target registers for each classical dataset to be loaded.
            Must be consistent with `data_or_shape` s.t. `len(data_or_shape) == len(target_shapes)`
            and `data_or_shape[-len(target_shapes[i]):] == target_shapes[i]`.
        target_bitsizes: Bitsize (or qdtype) of the target registers for each classical
            dataset to be loaded. This can be deduced from the maximum element of each
            of the datasets. Must be consistent with `data_or_shape` s.t.
            `len(data_or_shape) == len(target_bitsizes)` and
            `target_bitsizes[i] >= max(data[i]).bitsize`.
        num_controls: The number of controls to instanstiate a controlled version of this bloq.
    """

    data_or_shape: Tuple[Union[NDArray, Shaped], ...] = attrs.field(
        converter=lambda x: tuple(np.array(y) if isinstance(y, (list, tuple)) else y for y in x),
        eq=_data_or_shape_to_tuple,
    )
    selection_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    target_bitsizes: Tuple[SymbolicInt, ...] = attrs.field(
        converter=lambda x: tuple(x.tolist() if isinstance(x, np.ndarray) else x)
    )
    target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = attrs.field(
        converter=lambda x: tuple(tuple(y) for y in x)
    )
    num_controls: SymbolicInt = 0

    def is_symbolic(self) -> bool:
        return is_symbolic(
            self.num_controls,
            *self.data_or_shape,
            *self.selection_bitsizes,
            *self.target_bitsizes,
            *[sh for target_shape in self.target_shapes for sh in target_shape],
        )

    @target_shapes.default
    def _default_target_shapes(self):
        return ((),) * len(self.data_or_shape)

    @cached_property
    def data_shape(self) -> Tuple[SymbolicInt, ...]:
        ret: Tuple[SymbolicInt, ...] = ()
        for data_or_shape, target_shape in zip(self.data_or_shape, self.target_shapes):
            data_shape = shape(data_or_shape)
            if target_shape:
                data_shape = data_shape[: -len(target_shape)]
            if ret == ():
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
        return cast(Tuple[np.ndarray, ...], self.data_or_shape)

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
        cls: Type[QROM_T],
        *data: ArrayLike,
        target_bitsizes: Optional[Union[SymbolicInt, Tuple[SymbolicInt, ...]]] = None,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QROM_T:
        _data = [np.array(d, dtype=int) for d in data]
        if target_bitsizes is None:
            target_bitsizes = tuple(max(int(np.max(d)).bit_length(), 1) for d in data)
        assert isinstance(target_bitsizes, tuple)  # Make mypy happy.
        if target_shapes == ():
            target_shapes = ((),) * len(target_bitsizes)
        selection_len = len(_data[0].shape) - len(target_shapes[0])
        selection_bitsizes = tuple((s - 1).bit_length() for s in _data[0].shape[:selection_len])
        return cls(
            data_or_shape=_data,
            selection_bitsizes=selection_bitsizes,
            target_bitsizes=target_bitsizes,
            target_shapes=target_shapes,
            num_controls=num_controls,
        )

    def with_data(self: QROM_T, *data: ArrayLike) -> QROM_T:
        _data = tuple([np.array(d, dtype=int) for d in data])
        assert all(shape(d1) == shape(d2) for d1, d2 in zip(_data, self.data_or_shape))
        assert len(_data) == len(self.target_bitsizes)
        assert all(t >= int(np.max(d)).bit_length() for t, d in zip(self.target_bitsizes, _data))
        return attrs.evolve(self, data_or_shape=_data)

    @classmethod
    def _build_from_bitsize(
        cls: Type[QROM_T],
        data_len_or_shape: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        target_bitsizes: Union[SymbolicInt, Tuple[SymbolicInt, ...]],
        *,
        target_shapes: Tuple[Tuple[SymbolicInt, ...], ...] = (),
        selection_bitsizes: Tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QROM_T:
        data_shape: Tuple[SymbolicInt, ...] = (
            (data_len_or_shape,)
            if isinstance(data_len_or_shape, (int, numbers.Number, sympy.Basic))
            else data_len_or_shape
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
            BQUInt(sb, l)
            for l, sb in zip(self.data_shape, self.selection_bitsizes)
            if is_symbolic(l) or l > 1
        ]
        if len(types) == 1:
            return (Register('selection', types[0]),)
        return tuple(Register(f'selection{i}', qdtype) for i, qdtype in enumerate(types))

    @cached_property
    def _target_reg_side(self) -> Side:
        return Side.THRU

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return tuple(
            Register(f'target{i}_', QAny(l), shape=sh, side=self._target_reg_side)
            for i, (l, sh) in enumerate(zip(self.target_bitsizes, self.target_shapes))
            if is_symbolic(l) or l
        )

    def on_classical_vals(
        self, **vals: Union['sympy.Symbol', 'ClassicalValT']
    ) -> Dict[str, 'ClassicalValT']:
        if not self.has_data():
            raise NotImplementedError(f'Symbolic {self} does not support classical simulation')
        vals = cast(Dict[str, 'ClassicalValT'], vals)
        if self.num_controls > 0:
            control = vals['control']
            if control != 2**self.num_controls - 1:
                return vals
            controls = {'control': control}
        else:
            controls = {}

        n_dim = len(self.selection_registers)
        if n_dim == 0:
            idx: Union[int, Tuple[int, ...]] = 0
            selections = {}
        elif n_dim == 1:
            idx = int(vals.pop('selection', 0))
            selections = {'selection': idx}
        else:
            # Multidimensional
            idx = tuple(int(vals[f'selection{i}']) for i in range(n_dim))
            selections = {f'selection{i}': idx[i] for i in range(n_dim)}  # type: ignore[index]

        # Retrieve the data; bitwise add them in to the input target values
        targets = {f'target{d_i}_': d[idx] for d_i, d in enumerate(self.data)}
        targets = {k: v ^ vals.get(k, 0) for k, v in targets.items()}
        if not (self._target_reg_side & Side.RIGHT):
            for reg_name, reg_val in targets.items():
                if np.any(reg_val):
                    raise ValueError(
                        f"Target register {reg_name} must be uncomputed before de-allocation. Found values {reg_val}"
                    )
            targets = {}
        return controls | selections | targets


_QROM_BASE_DOC = BloqDocSpec(bloq_cls=QROMBase, import_line='', examples=[])
