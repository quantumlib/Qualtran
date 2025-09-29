from typing import Union, Sequence
from qualtran.symbolics import is_symbolic, SymbolicInt
from qualtran import QGF

from galois import Poly

def qgf_converter(x: Union[QGF, int, Poly, SymbolicInt, Sequence[int]]) -> QGF:
    if isinstance(x, QGF):
        return x
    if isinstance(x, int):
        return QGF(2, x)
    if is_symbolic(x):
        return QGF(2, x)
    if isinstance(x, Poly):
        return QGF(2, x.degree, x)
    p = Poly.Degrees(x)
    return QGF(2, p.degree, p)

