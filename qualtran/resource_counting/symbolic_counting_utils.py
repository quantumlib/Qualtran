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
from typing import cast, Sized, Union

import numpy as np
import sympy
from attrs import field, frozen, validators
from cirq._doc import document

SymbolicFloat = Union[float, sympy.Expr]
document(SymbolicFloat, """A floating point value or a sympy expression.""")

SymbolicInt = Union[int, sympy.Expr]
document(SymbolicFloat, """A floating point value or a sympy expression.""")

SymbolicComplex = Union[complex, sympy.Expr]
document(SymbolicComplex, """A complex value or a sympy expression.""")


@frozen
class Shaped:
    """Symbolic value for an object that has a shape.

    A Shaped object can be used as a symbolic replacement for any object that has an attribute `shape`,
    for example numpy NDArrays.
    Each dimension can be either an positive integer value or a sympy expression.

    This is useful to do symbolic analysis of Bloqs whose call graph only depends on the shape of the input,
    but not on the actual values.
    For example, T-cost of the `QROM` Bloq depends only on the iteration length (shape) and not on actual data values.
    """

    shape: tuple[SymbolicInt, ...] = field(validator=validators.instance_of(tuple))

    def is_symbolic(self):
        return True


def is_symbolic(*args) -> bool:
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


def pi(*args) -> SymbolicFloat:
    return sympy.pi if is_symbolic(*args) else np.pi


def log2(x: SymbolicFloat) -> SymbolicFloat:
    from sympy.codegen.cfunctions import log2

    if not isinstance(x, sympy.Basic):
        return np.log2(x)
    return log2(x)


def sabs(x: SymbolicFloat) -> SymbolicFloat:
    return cast(SymbolicFloat, abs(x))


def ceil(x: SymbolicFloat) -> SymbolicInt:
    if not isinstance(x, sympy.Basic):
        return int(np.ceil(x))
    return sympy.ceiling(x)


def floor(x: SymbolicFloat) -> SymbolicInt:
    if not isinstance(x, sympy.Basic):
        return int(np.floor(x))
    return sympy.floor(x)


def bit_length(x: SymbolicFloat) -> SymbolicInt:
    """Returns the number of bits required to represent the integer part of positive float `x`."""
    if not is_symbolic(x) and 0 <= x < 1:
        return 0
    ret = ceil(log2(x))
    if is_symbolic(ret):
        return ret
    return ret + 1 if ret == floor(log2(x)) else ret


def smax(*args):
    if any(isinstance(arg, sympy.Basic) for arg in args):
        return sympy.Max(*args)
    return max(*args)


def smin(*args):
    if is_symbolic(*args):
        return sympy.Min(*args)
    return min(*args)


def acos(x: SymbolicFloat) -> SymbolicFloat:
    if not isinstance(x, sympy.Basic):
        return np.arccos(x)
    return sympy.acos(x)


def sconj(x: SymbolicComplex) -> SymbolicComplex:
    """Compute the complex conjugate."""
    return sympy.conjugate(x) if isinstance(x, sympy.Expr) else np.conjugate(x)


def slen(x: Union[Sized, Shaped]) -> SymbolicInt:
    if isinstance(x, Shaped):
        return x.shape[0]
    return len(x)
