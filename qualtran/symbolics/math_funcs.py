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
from typing import cast, overload, Sized, Tuple, Union

import numpy as np
import sympy

from qualtran.symbolics.types import (
    HasLength,
    is_symbolic,
    Shaped,
    SymbolicComplex,
    SymbolicFloat,
    SymbolicInt,
)


def pi(*args) -> SymbolicFloat:
    return sympy.pi if is_symbolic(*args) else np.pi


def log2(x: SymbolicFloat) -> SymbolicFloat:
    from sympy.codegen.cfunctions import log2

    if not isinstance(x, sympy.Basic):
        return np.log2(x)
    return log2(x)


def sabs(x: SymbolicFloat) -> SymbolicFloat:
    return cast(SymbolicFloat, abs(x))


def ssqrt(x: SymbolicFloat) -> SymbolicFloat:
    if isinstance(x, sympy.Basic):
        return sympy.sqrt(x)
    return np.sqrt(x)


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


def prod(*args: SymbolicInt) -> SymbolicInt:
    ret: SymbolicInt = 1
    for arg in args:
        ret = ret * arg
    return ret


def acos(x: SymbolicFloat) -> SymbolicFloat:
    if not isinstance(x, sympy.Basic):
        return np.arccos(x)
    return sympy.acos(x)


def sconj(x: SymbolicComplex) -> SymbolicComplex:
    """Compute the complex conjugate."""
    return sympy.conjugate(x) if isinstance(x, sympy.Expr) else np.conjugate(x)


def slen(x: Union[Sized, Shaped, HasLength]) -> SymbolicInt:
    if isinstance(x, Shaped):
        return x.shape[0]
    if isinstance(x, HasLength):
        return x.n
    return len(x)


@overload
def shape(x: np.ndarray) -> Tuple[int, ...]:
    ...


@overload
def shape(x: Shaped) -> Tuple[SymbolicInt, ...]:
    ...


def shape(x: Union[np.ndarray, Shaped]):
    return x.shape
