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

"""Contrived test bloqs that exercise multi-dimensional quantum variable arrays.

These bloqs exist to test that Qualtran's infrastructure correctly handles registers
with ``shape`` of dimensionality >= 2 (i.e. ndarrays of soquets).  All operations
are classically simulable (X, CNOT, Toffoli) so ``call_classically`` can be used for
verification.
"""

from functools import cached_property
from typing import Dict

import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, QBit, Register, Signature, Soquet, SoquetT
from qualtran.bloqs.basic_gates import CNOT, XGate
from qualtran.bloqs.basic_gates.toffoli import Toffoli

from qualtran.simulation.classical_sim import ClassicalValT


@frozen
class TestNDGrid(Bloq):
    """A test bloq with a 2-D grid register and asymmetric classical logic.

    This bloq is designed to exercise multi-dimensional (nd >= 2) quantum
    variable arrays through ``build_composite_bloq`` wiring.  It uses only
    classically-simulable gates (X, CNOT, Toffoli) so the result can be
    verified with ``call_classically``.

    The operations performed break all row/column/transpose symmetry of the
    2-D ``grid`` register so that bugs hiding behind accidental symmetry are
    caught.

    Registers:
        grid: A 3×2 matrix of single-qubit registers (``QBit()``, shape ``(3, 2)``).
        ctrl: A 1-D array of two control qubits (``QBit()``, shape ``(2,)``).
        flag: A single-qubit flag register (``QBit()``).

    Decomposition (step by step):
        1. ``X`` on ``grid[0, 0]``  – unconditional flip of one corner.
        2. ``CNOT(ctrl=grid[0, 1], target=grid[1, 0])``
           – row-0-col-1 controls row-1-col-0 (asymmetric in rows *and* cols).
        3. ``CNOT(ctrl=ctrl[0], target=grid[1, 1])``
           – mixes the 1-D ``ctrl`` array into the 2-D grid.
        4. ``Toffoli(ctrl=[grid[2, 0], ctrl[1]], target=grid[2, 1])``
           – two controls from *different* registers, target in the grid.
        5. ``CNOT(ctrl=grid[2, 1], target=flag)``
           – propagates the grid result into the scalar ``flag``.
        6. ``CNOT(ctrl=grid[1, 0], target=grid[0, 1])``
           – feeds information *back up* the grid (row 1 → row 0).
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('grid', QBit(), shape=(3, 2)),
                Register('ctrl', QBit(), shape=(2,)),
                Register('flag', QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        grid: 'SoquetT',
        ctrl: 'SoquetT',
        flag: Soquet,
    ) -> Dict[str, 'SoquetT']:
        # Step 1: X on grid[0, 0]
        grid[0, 0] = bb.add(XGate(), q=grid[0, 0])

        # Step 2: CNOT  ctrl=grid[0,1]  target=grid[1,0]
        grid[0, 1], grid[1, 0] = bb.add(CNOT(), ctrl=grid[0, 1], target=grid[1, 0])

        # Step 3: CNOT  ctrl=ctrl[0]  target=grid[1,1]
        ctrl[0], grid[1, 1] = bb.add(CNOT(), ctrl=ctrl[0], target=grid[1, 1])

        # Step 4: Toffoli  ctrl=[grid[2,0], ctrl[1]]  target=grid[2,1]
        (grid[2, 0], ctrl[1]), grid[2, 1] = bb.add(
            Toffoli(), ctrl=np.array([grid[2, 0], ctrl[1]]), target=grid[2, 1]
        )

        # Step 5: CNOT  ctrl=grid[2,1]  target=flag
        grid[2, 1], flag = bb.add(CNOT(), ctrl=grid[2, 1], target=flag)

        # Step 6: CNOT  ctrl=grid[1,0]  target=grid[0,1]
        grid[1, 0], grid[0, 1] = bb.add(CNOT(), ctrl=grid[1, 0], target=grid[0, 1])

        return {'grid': grid, 'ctrl': ctrl, 'flag': flag}

    def on_classical_vals(
        self,
        grid: NDArray[np.integer],
        ctrl: NDArray[np.integer],
        flag: int,
    ) -> Dict[str, ClassicalValT]:
        # Work on copies so we don't mutate the caller's arrays.
        g = grid.copy()
        c = ctrl.copy()
        f = int(flag)

        # Step 1
        g[0, 0] ^= 1
        # Step 2
        g[1, 0] ^= g[0, 1]
        # Step 3
        g[1, 1] ^= c[0]
        # Step 4
        g[2, 1] ^= g[2, 0] & c[1]
        # Step 5
        f ^= g[2, 1]
        # Step 6
        g[0, 1] ^= g[1, 0]

        return {'grid': g, 'ctrl': c, 'flag': f}


@frozen
class TestND3Grid(Bloq):
    """A test bloq with a 3-D register to exercise true rank-3 nd-arrays.

    Registers:
        cube: A 2×2×2 cube of single-qubit registers (``QBit()``, shape ``(2, 2, 2)``).
        aux: A scalar qubit (``QBit()``).

    Decomposition:
        1. ``X`` on ``cube[0, 0, 0]``
        2. ``CNOT(ctrl=cube[0, 0, 1], target=cube[0, 1, 0])``
        3. ``CNOT(ctrl=cube[0, 1, 1], target=cube[1, 0, 0])``
        4. ``Toffoli(ctrl=[cube[1, 0, 0], cube[1, 0, 1]], target=cube[1, 1, 0])``
        5. ``CNOT(ctrl=cube[1, 1, 0], target=cube[1, 1, 1])``
        6. ``CNOT(ctrl=cube[1, 1, 1], target=aux)``
    """

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register('cube', QBit(), shape=(2, 2, 2)),
                Register('aux', QBit()),
            ]
        )

    def build_composite_bloq(
        self,
        bb: 'BloqBuilder',
        cube: 'SoquetT',
        aux: Soquet,
    ) -> Dict[str, 'SoquetT']:
        # Step 1
        cube[0, 0, 0] = bb.add(XGate(), q=cube[0, 0, 0])
        # Step 2
        cube[0, 0, 1], cube[0, 1, 0] = bb.add(
            CNOT(), ctrl=cube[0, 0, 1], target=cube[0, 1, 0]
        )
        # Step 3
        cube[0, 1, 1], cube[1, 0, 0] = bb.add(
            CNOT(), ctrl=cube[0, 1, 1], target=cube[1, 0, 0]
        )
        # Step 4
        (cube[1, 0, 0], cube[1, 0, 1]), cube[1, 1, 0] = bb.add(
            Toffoli(), ctrl=np.array([cube[1, 0, 0], cube[1, 0, 1]]), target=cube[1, 1, 0]
        )
        # Step 5
        cube[1, 1, 0], cube[1, 1, 1] = bb.add(
            CNOT(), ctrl=cube[1, 1, 0], target=cube[1, 1, 1]
        )
        # Step 6
        cube[1, 1, 1], aux = bb.add(CNOT(), ctrl=cube[1, 1, 1], target=aux)

        return {'cube': cube, 'aux': aux}

    def on_classical_vals(
        self,
        cube: NDArray[np.integer],
        aux: int,
    ) -> Dict[str, ClassicalValT]:
        c = cube.copy()
        a = int(aux)

        # Step 1
        c[0, 0, 0] ^= 1
        # Step 2
        c[0, 1, 0] ^= c[0, 0, 1]
        # Step 3
        c[1, 0, 0] ^= c[0, 1, 1]
        # Step 4
        c[1, 1, 0] ^= c[1, 0, 0] & c[1, 0, 1]
        # Step 5
        c[1, 1, 1] ^= c[1, 1, 0]
        # Step 6
        a ^= c[1, 1, 1]

        return {'cube': c, 'aux': a}
