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

import qualtran.rotation_synthesis.matrix._clifford_t_repr as ctr
import qualtran.rotation_synthesis.matrix._su2_ct as _su2_ct


def _make_random_su(n: int, m: int, random_cliffords: bool = False, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    gates = [_su2_ct.Tx, _su2_ct.Ty, _su2_ct.Tz]
    if random_cliffords:
        gates += [_su2_ct.SSqrt2, _su2_ct.HSqrt2]
    for _ in range(n):
        res = _su2_ct.ISqrt2
        for i in rng.choice(len(gates), m):
            res = res @ gates[i]
        yield res


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_xyz_seq(g):
    seq = ctr.to_sequence(g, 'xyz')
    assert not any('*' in g for g in seq)
    got = _su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
def test_to_xz_seq(g: _su2_ct.SU2CliffordT):
    g = g.rescale()
    seq = ctr.to_sequence(g, 'xz')
    assert not any('Ty' in g for g in seq)
    first_t = None
    for i in range(len(seq)):
        if seq[i].startswith('T'):
            first_t = i
            break
    if first_t is not None:
        ts = 'Tx', 'Tx*', 'Tz', 'Tz*'
        assert all(s in ts for s in seq[first_t:-2]), f'{seq=}'
    got = _su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g


@pytest.mark.parametrize("g", _make_random_su(50, 5, random_cliffords=True, seed=0))
@pytest.mark.parametrize("fmt", ('xyz', 'xz', 't'))
def test_to_cirq(g: _su2_ct.SU2CliffordT, fmt: str):
    g = g.rescale()
    circuit = cirq.Circuit(ctr.to_cirq(g, fmt))
    unitary = cirq.unitary(circuit)
    u = g.matrix.astype(complex)
    u = u / np.linalg.det(u) ** 0.5
    assert cirq.protocols.equal_up_to_global_phase(u, unitary)


@pytest.mark.parametrize(
    ["g", "fmt", "expected"],
    [
        [_su2_ct.HSqrt2, "t", ('"Z^½"', '"X"', '"Z^½"', '"X"', '"H"')],
        [_su2_ct.SSqrt2, "t", ('{"id":"Rzft","arg":"pi/2"}',)],
        [_su2_ct.Tz, "t", ('{"id":"Rzft","arg":"pi/4"}',)],
    ],
)
def test_to_quirk(g: _su2_ct.SU2CliffordT, fmt: str, expected: tuple[str, ...]):
    assert ctr.to_quirk(g, fmt) == expected


@pytest.mark.parametrize("g", _make_random_su(50, 10, random_cliffords=True, seed=0))
def test_to_matsumoto_amano_seq(g: _su2_ct.SU2CliffordT):
    g = g.rescale()
    seq = ctr.to_sequence(g, 't')
    prev_t = -1
    # Check that the reversed list matches the regular expression $(T|\eps)(HT|SHT)^*C$.
    seq_r = list(reversed(seq))
    for i in range(len(seq_r)):
        assert seq_r[i] in ('T', 'H', 'S', 'Z', 'X')
        if i == 0 and seq_r[0] == 'T':
            prev_t = 0
            continue
        if seq_r[i] == 'T':
            # Check that this is one of HT, SHT syllabes
            assert prev_t in (i - 2, i - 3)
            if prev_t == i - 2:
                assert seq_r[i - 1] == 'H'
            else:
                assert seq_r[i - 2] + seq_r[i - 1] == 'SH'
            prev_t = i
    got = _su2_ct.SU2CliffordT.from_sequence(seq)
    assert got == g
