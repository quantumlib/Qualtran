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
from typing import overload, Sized, TypeVar, Union

import numpy as np
import sympy
from attrs import field, frozen, validators
from typing_extensions import TypeIs

SymbolicFloat = Union[float, sympy.Expr]
"""A floating point value or a sympy expression."""

SymbolicInt = Union[int, sympy.Expr]
"""An integer value or a sympy expression."""

SymbolicComplex = Union[complex, sympy.Expr]
"""A complex value or a sympy expression."""


@frozen
class Shaped:
    """Symbolic value for an object that has a shape.

    A Shaped object can be used as a symbolic replacement for any object that has an
    attribute `shape`, for example numpy `NDArrays`. Each dimension can be either
    a positive integer value or a sympy expression.

    For the symbolic variant of a tuple or sequence of values, see `HasLength`.

    This is useful to do symbolic analysis of Bloqs whose call graph only depends on the shape
    of the input, but not on the actual values. For example, T-cost of the `QROM` Bloq depends
    only on the iteration length (shape) and not on actual data values. In this case, for the
    bloq attribute `data`, we can use the type:

    ```py
    data: Union[NDArray, Shaped]
    ```
    """

    shape: tuple[SymbolicInt, ...] = field(validator=validators.instance_of(tuple))

    def is_symbolic(self):
        return True


@frozen
class HasLength:
    """Symbolic value for an object that has a length.

    This is used as a "symbolic" tuple. The length can either be a positive integer
    or a sympy expression. For example, if a bloq attribute is a tuple of ints,
    we can use the type:

    ```py
    values: Union[tuple, HasLength]
    ```

    For the symbolic variant of a NDArray, see `Shaped`.

    Note that we cannot override __len__ and return a sympy symbol because Python has
    special treatment for __len__ and expects you to return a non-negative integers.

    See https://docs.python.org/3/reference/datamodel.html#object.__len__ for more details.
    """

    n: SymbolicInt

    def is_symbolic(self):
        return True


@overload
def slen(x: Sized) -> int: ...


@overload
def slen(x: Union[Shaped, HasLength]) -> sympy.Expr: ...


def slen(x: Union[Sized, Shaped, HasLength]) -> SymbolicInt:
    if isinstance(x, Shaped):
        return x.shape[0]
    if isinstance(x, HasLength):
        return x.n
    return len(x)


@overload
def shape(x: np.ndarray) -> tuple[int, ...]: ...


@overload
def shape(x: Shaped) -> tuple[SymbolicInt, ...]: ...


def shape(x: Union[np.ndarray, Shaped]):
    return x.shape


T = TypeVar('T')


@overload
def is_symbolic(
    arg: Union[T, sympy.Expr, Shaped, HasLength], /
) -> TypeIs[Union[sympy.Expr, Shaped, HasLength]]: ...


@overload
def is_symbolic(*args) -> bool: ...


def is_symbolic(*args) -> Union[TypeIs[Union[sympy.Expr, Shaped, HasLength]], bool]:
    """Returns whether the inputs contain any symbolic object.

    Returns:
        True if any argument is either a sympy object,
        or implements the `is_symbolic` method which returns True.
    """

    if len(args) != 1:
        return any(is_symbolic(arg) for arg in args)

    (arg,) = args
    if isinstance(arg, sympy.Basic):
        return True

    checker = getattr(arg, 'is_symbolic', None)
    if checker is not None:
        return checker()

    return False
