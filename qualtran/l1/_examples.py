#  Copyright 2026 Google LLC
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
"""A diverse set of example bloqs for Qualtran-L1.

These examples are drawn from the `qualtran.bloqs` standard library and chosen to
exercise a broad range of Qualtran-L1 features: atomic (externed) bloqs,
decomposed (`qdef`) bloqs, bookkeeping (`qcast`) bloqs, shaped registers,
LEFT/RIGHT (allocating/freeing) registers, and nested subroutine calls.

The list is used by:
 - `qualtran/l1/_roundtrip_test.py` to assert that each example round-trips.
 - `dev_tools/generate-l1-reference.py` to (re)generate the
   committed `.qlt` reference files under `qualtran/l1/examples/`.

Each example uses a lazily-evaluated factory so that importing this module is
cheap and does not construct (potentially expensive) bloqs eagerly.
"""

from __future__ import annotations

import pathlib
from typing import Callable, List, TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    import qualtran as qlt


@attrs.frozen
class L1Example:
    """A named, lazily-constructed example bloq.

    Attributes:
        name: A short, filesystem-safe identifier. Used as the `.qlt` filename stem.
        factory: A zero-argument callable that constructs the bloq.
        slow: Whether constructing/round-tripping this bloq is slow (the
            corresponding pytest is marked `pytest.mark.slow`).
        docstring: A one-line description of what the example demonstrates.
    """

    name: str
    factory: Callable[[], 'qlt.Bloq'] = attrs.field(repr=False)
    slow: bool = False
    docstring: str = ''

    def make(self) -> 'qlt.Bloq':
        """Construct the bloq."""
        return self.factory()

    def reference_path(self) -> pathlib.Path:
        """The path to this example's committed `.qlt` reference file."""
        return reference_path(self.name)


def _cnot() -> 'qlt.Bloq':
    from qualtran.bloqs.basic_gates import CNOT

    return CNOT()


def _cswap() -> 'qlt.Bloq':
    from qualtran.bloqs.basic_gates import CSwap

    return CSwap(bitsize=5)


def _hadamard() -> 'qlt.Bloq':
    from qualtran.bloqs.basic_gates import Hadamard

    return Hadamard()


def _rz() -> 'qlt.Bloq':
    from qualtran.bloqs.basic_gates import Rz

    return Rz(0.25)


def _qft_textbook() -> 'qlt.Bloq':
    from qualtran.bloqs.qft import QFTTextBook

    return QFTTextBook(3)


def _test_nd3grid() -> 'qlt.Bloq':
    from qualtran.bloqs.for_testing import TestND3Grid

    return TestND3Grid()


def _multi_and() -> 'qlt.Bloq':
    from qualtran.bloqs.mcmt import MultiAnd

    return MultiAnd(cvs=(1, 1, 0, 1))


def _and_bloq() -> 'qlt.Bloq':
    from qualtran.bloqs.mcmt import And

    return And(1, 1)


def _multi_target_cnot() -> 'qlt.Bloq':
    from qualtran.bloqs.mcmt import MultiTargetCNOT

    return MultiTargetCNOT(4)


def _negate() -> 'qlt.Bloq':
    import qualtran.dtype as qdt
    from qualtran.bloqs.arithmetic import Negate

    return Negate(qdt.QInt(8))


def _add() -> 'qlt.Bloq':
    import qualtran.dtype as qdt
    from qualtran.bloqs.arithmetic import Add

    return Add(qdt.QUInt(4))


def _bitwise_not() -> 'qlt.Bloq':
    import qualtran.dtype as qdt
    from qualtran.bloqs.arithmetic import BitwiseNot

    return BitwiseNot(qdt.QUInt(4))


def _select_hubbard() -> 'qlt.Bloq':
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import SelectHubbard

    return SelectHubbard(x_dim=2, y_dim=2)


def _multi_control_x() -> 'qlt.Bloq':
    from qualtran.bloqs.mcmt import MultiControlX

    return MultiControlX(cvs=(1, 0, 1))


def _qrom() -> 'qlt.Bloq':
    from qualtran.bloqs.data_loading.qrom import QROM

    return QROM.build_from_data([1, 2, 3, 4])


L1_EXAMPLES: List[L1Example] = [
    L1Example('cnot', _cnot, docstring='An atomic two-qubit gate (externed leaf).'),
    L1Example(
        'and_bloq',
        _and_bloq,
        docstring='A logical-AND that allocates its target (a RIGHT-only register).',
    ),
    L1Example('cswap', _cswap, docstring='A controlled swap decomposing into TwoBitCSwaps.'),
    L1Example(
        'hadamard',
        _hadamard,
        docstring='The Hadamard gate; a genuinely quantum gate that creates superposition (atomic).',
    ),
    L1Example(
        'rz',
        _rz,
        docstring='A single-qubit Z-rotation parameterized by a real angle; exercises float args.',
    ),
    L1Example(
        'qft_textbook',
        _qft_textbook,
        docstring='The textbook quantum Fourier transform on 3 qubits; a quantum algorithm '
        'decomposing into Hadamards and controlled phase rotations.',
    ),
    L1Example(
        'test_nd3grid',
        _test_nd3grid,
        docstring='A rank-3 (2x2x2) ndarray of qubit registers; exercises '
        'multi-dimensional quantum-variable arrays.',
    ),
    L1Example(
        'multi_and',
        _multi_and,
        docstring='A multi-controlled AND with mixed control values; decomposes into Ands.',
    ),
    L1Example(
        'multi_target_cnot',
        _multi_target_cnot,
        docstring='A CNOT with one control fanned out to many targets.',
    ),
    L1Example('negate', _negate, docstring='Two\'s-complement negation of a signed integer.'),
    L1Example('add', _add, docstring='Out-of-place-free integer addition on unsigned integers.'),
    L1Example(
        'bitwise_not',
        _bitwise_not,
        docstring='Bitwise NOT; decomposes via Split/Join (qcast) around X gates.',
    ),
    L1Example(
        'multi_control_x',
        _multi_control_x,
        docstring='A multi-controlled X; decomposes via the `Adjoint` meta-bloq.',
    ),
    L1Example(
        'qrom', _qrom, docstring='A QROM data-loader; decomposes via the `Controlled` meta-bloq.'
    ),
    L1Example(
        'select_hubbard',
        _select_hubbard,
        slow=True,
        docstring='The SELECT oracle for the 2D Hubbard model (a large, nested example).',
    ),
]


def get_l1_examples(include_slow: bool = True) -> List[L1Example]:
    """Return the curated example bloqs.

    Args:
        include_slow: If `False`, examples marked `slow` are omitted.
    """
    if include_slow:
        return list(L1_EXAMPLES)
    return [ex for ex in L1_EXAMPLES if not ex.slow]


def reference_dir() -> pathlib.Path:
    """The directory holding the committed `.qlt` reference files."""
    return pathlib.Path(__file__).parent / 'examples'


def reference_path(name: str) -> pathlib.Path:
    """The path to the `.qlt` reference file for the example named `name`."""
    return reference_dir() / f'{name}.qlt'
