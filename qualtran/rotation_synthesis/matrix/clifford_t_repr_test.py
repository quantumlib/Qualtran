#  Copyright 2025 Google LLC
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

from typing import Optional

import cirq
import numpy as np
import pytest

import qualtran.rotation_synthesis.matrix.clifford_t_repr as ctr
import qualtran.rotation_synthesis.matrix.su2_ct as su2_ct


def _make_random_su(n: int, m: int, random_cliffords: bool = False, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    gates = [su2_ct.Tx, su2_ct.Ty, su2_ct.Tz]
    if random_cliffords:
        gates += [su2_ct.SSqrt2, su2_ct.HSqrt2]
    for _ in range(n):
        res = su2_ct.ISqrt2
        for i in rng.choice(len(gates), m):
            res = res @ gates[i]
        yield res


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_xyz_seq(g):
    seq = ctr.to_sequence(g, 'xyz')
    assert not any('*' in g for g in seq)
    got = su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_xz_seq(g):
    seq = ctr.to_sequence(g, 'xz')
    assert not any('Ty' in g for g in seq)
    got = su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_cirq(g):
    circuit = cirq.Circuit(ctr.to_cirq(g, 'xyz'))
    unitary = cirq.unitary(circuit)
    u = g.matrix.astype(complex)
    u = u / np.linalg.det(u) ** 0.5
    assert cirq.protocols.equal_up_to_global_phase(u, unitary)
