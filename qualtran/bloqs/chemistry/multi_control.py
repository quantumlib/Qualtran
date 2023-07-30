import itertools
from functools import cached_property
from typing import Any, Dict, Optional, Set, Tuple, Union

import numpy as np
import quimb.tensor as qtn
import sympy
from attrs import field, frozen
from numpy.typing import NDArray

from qualtran import Bloq, Register, Side, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import TGate
from qualtran.bloqs.util_bloqs import ArbitraryClifford
from qualtran.drawing import Circle, directional_text_box, WireSymbol
from qualtran.resource_counting import big_O, SympySymbolAllocator


@frozen
class MultiControl(Bloq):
    """A multi-controlled gate.

    Args:
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.

    Registers:
     - ctrl: An n-bit control register.
    """

    bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...]

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register(f'ctrl_{i}', bitsize=bs) for i, bs in enumerate(self.bitsizes)])
