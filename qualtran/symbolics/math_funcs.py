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
from typing import cast, Iterable, overload, Sized, Tuple, TypeVar, Union

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


@overload
def log2(x: float) -> float:
    ...


@overload
def log2(x: sympy.Expr) -> sympy.Expr:
    ...


def log2(x: SymbolicFloat) -> SymbolicFloat:
    from sympy.codegen.cfunctions import log2

    if not is_symbolic(x):
        return np.log2(x)
    return log2(x)


@overload
def ln(x: float) -> float:
    ...


@overload
def ln(x: sympy.Expr) -> sympy.Expr:
    ...


def ln(x: SymbolicFloat) -> SymbolicFloat:
    from sympy.codegen.cfunctions import log

    if not is_symbolic(x):
        return np.log(x)
    return log(x)


@overload
def sexp(x: complex) -> complex:
    ...


@overload
def sexp(x: sympy.Expr) -> sympy.Expr:
    ...


def sexp(x: SymbolicComplex) -> SymbolicComplex:
    if is_symbolic(x):
        return sympy.exp(x)
    return np.exp(x)


@overload
def sarg(x: complex) -> float:
    ...


@overload
def sarg(x: sympy.Expr) -> sympy.Expr:
    ...


def sarg(x: SymbolicComplex) -> SymbolicFloat:
    r"""Argument $t$ of a complex number $r e^{i t}$"""
    if is_symbolic(x):
        return sympy.arg(x)
    return float(np.angle(x))


@overload
def sabs(x: float) -> float:
    ...


@overload
def sabs(x: sympy.Expr) -> sympy.Expr:
    ...


def sabs(x: SymbolicFloat) -> SymbolicFloat:
    return cast(SymbolicFloat, abs(x))


@overload
def ssqrt(x: float) -> float:
    ...


@overload
def ssqrt(x: sympy.Expr) -> sympy.Expr:
    ...


def ssqrt(x: SymbolicFloat) -> SymbolicFloat:
    if is_symbolic(x):
        return sympy.sqrt(x)
    return np.sqrt(x)


@overload
def ceil(x: float) -> int:
    ...


@overload
def ceil(x: sympy.Expr) -> sympy.Expr:
    ...


def ceil(x: SymbolicFloat) -> SymbolicInt:
    if not is_symbolic(x):
        return int(np.ceil(x))
    return sympy.ceiling(x)


@overload
def floor(x: float) -> int:
    ...


@overload
def floor(x: sympy.Expr) -> sympy.Expr:
    ...


def floor(x: SymbolicFloat) -> SymbolicInt:
    if not is_symbolic(x):
        return int(np.floor(x))
    return sympy.floor(x)


@overload
def bit_length(x: float) -> int:
    ...


@overload
def bit_length(x: sympy.Expr) -> sympy.Expr:
    ...


def bit_length(x: SymbolicFloat) -> SymbolicInt:
    """Returns the number of bits required to represent the integer part of positive float `x`."""
    if not is_symbolic(x) and 0 <= x < 1:
        return 0
    return ceil(log2(floor(x) + 1))


def smax(*args):
    """Returns the maximum of the given arguments, which may be symbolic.

    Args:
        args: Either a pack of arguments or a single Iterable of arguments.
              At least one argument must be provided in this pack or Iterable.

    Returns:
        The maximum of the given arguments.
    """
    if len(args) == 0:
        raise ValueError("smax expected at least 1 argument, got 0")
    if len(args) == 1:
        (it,) = args
        if isinstance(it, Iterable):
            args = tuple(arg for arg in it)
        if len(args) == 0:
            raise ValueError("smax() arg is an empty sequence")
    if len(args) == 1:
        (arg,) = args
        return arg
    if is_symbolic(*args):
        return sympy.Max(*args)
    return max(*args)


def smin(*args):
    """Returns the minimum of the given arguments, which may be symbolic.

    Args:
        args: Either a pack of arguments or a single Iterable of arguments.
              At least one argument must be provided in this pack or Iterable.

    Returns:
        The minimum of the given arguments.
    """
    if len(args) == 0:
        raise ValueError("smin expected at least 1 argument, got 0")
    if len(args) == 1:
        (it,) = args
        if isinstance(it, Iterable):
            args = tuple(arg for arg in it)
        if len(args) == 0:
            raise ValueError("smin() arg is an empty sequence")
    if len(args) == 1:
        (arg,) = args
        return arg
    if is_symbolic(*args):
        return sympy.Min(*args)
    return min(*args)


# This is only used in the type signature of functions that should be generic over different
# symbolic types, in situations where Union and @overload are not sufficient.
# The user should not need to invoke it directly. Rather, the user can use a function that
# takes SymbolicT by calling it with e.g. a SymbolicInt. Correspondingly, if the type signature
# of the function returns SymbolicT, then this call will then return a SymbolicInt.
SymbolicT = TypeVar('SymbolicT', SymbolicInt, SymbolicFloat, SymbolicComplex)


def prod(args: Iterable[SymbolicT]) -> SymbolicT:
    ret: SymbolicT = 1
    for arg in args:
        ret = ret * arg
    return ret


def ssum(args: Iterable[SymbolicT]) -> SymbolicT:
    ret: SymbolicT = 0
    for arg in args:
        ret = ret + arg
    return ret


@overload
def acos(x: float) -> float:
    ...


@overload
def acos(x: sympy.Expr) -> sympy.Expr:
    ...


def acos(x: SymbolicFloat) -> SymbolicFloat:
    if not is_symbolic(x):
        return np.arccos(x)
    return sympy.acos(x)


@overload
def sconj(x: complex) -> complex:
    ...


@overload
def sconj(x: sympy.Expr) -> sympy.Expr:
    ...


def sconj(x: SymbolicComplex) -> SymbolicComplex:
    """Compute the complex conjugate."""
    return sympy.conjugate(x) if is_symbolic(x) else np.conjugate(x)


@overload
def slen(x: Sized) -> int:
    ...


@overload
def slen(x: Union[Shaped, HasLength]) -> sympy.Expr:
    ...


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


def is_zero(x: SymbolicInt) -> bool:
    """check if a symbolic integer is zero

    If it returns True, then the value is definitely 0.
    If it returns False, then the value is either non-zero,
    or could not be symbolically symplified to a zero.
    """
    if is_symbolic(x):
        try:
            zero = x.equals(0)
        except ValueError:
            zero = False
        return zero

    return x == 0
