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

import math
from typing import cast, Mapping, Optional, Union

import cirq
import numpy as np

import qualtran.rotation_synthesis.matrix._generation as ctg
import qualtran.rotation_synthesis.matrix._su2_ct as _su2_ct
import qualtran.rotation_synthesis.rings._zsqrt2 as _zsqrt2

_TWO = _zsqrt2.ZSqrt2(2)


def clifford(matrix: _su2_ct.SU2CliffordT) -> tuple[str, ...]:
    assert matrix.det() == 2
    cliffords = ctg.generate_cliffords()
    if matrix in cliffords:
        return cliffords[matrix]
    return ("Z", "X", "Z", "X") + cliffords[-matrix]


def _xyz_sequence(matrix: _su2_ct.SU2CliffordT) -> tuple[str, ...]:
    seq = []
    while matrix.det() > _TWO:
        t = _su2_ct._key_map()[matrix._key]
        nxt = (_su2_ct.GATE_MAP[t].adjoint() @ matrix).scale_down()
        assert nxt is not None
        assert nxt.det() < matrix.det()
        seq.append(t)
        matrix = nxt
    return clifford(matrix) + tuple(seq[::-1])


_T_list = [
    ('Tz', _su2_ct.Tz),
    ('Tx', _su2_ct.Tx),
    ('Tz*', _su2_ct.Tz.adjoint()),
    ('Tx*', _su2_ct.Tx.adjoint()),
]


def _xz_sequence(
    matrix: _su2_ct.SU2CliffordT, use_hs: bool = True, prv: str = 'dummy'
) -> Optional[tuple[str, ...]]:
    if matrix.det() == 2:
        return clifford(matrix)
    cliffords = [_su2_ct.ISqrt2]
    if use_hs:
        cliffords.append(_su2_ct.HSqrt2)
        cliffords.append(_su2_ct.HSqrt2 @ _su2_ct.SSqrt2)
    pref = prv.removesuffix('*')
    for c in cliffords:
        for name, t in _T_list:
            if name.startswith(pref):
                continue
            new = c.adjoint() @ matrix
            new = t.adjoint() @ new
            new = new.scale_down()
            if new is None or not new.is_valid():
                continue
            seq = _xz_sequence(new, False, name)
            if seq is None:
                continue
            gates = cast(tuple[str, ...], c.gates)
            return seq + (name,) + gates
    return None


def _matsumoto_amano_syllable(matrix: _su2_ct.SU2CliffordT) -> list[str]:
    """Computes the next syllable in the Matsumoto-Amano decomposition for matrix.

    Returns:
        the next syllable as a list of gates, representing either T, HT, or SHT.

    Raises:
        ValueError if the parity matrix doesn't match any of the forms in
        Lemma 4.10, https://arxiv.org/abs/1312.6584.
    """
    parity = matrix.bloch_form_parity()
    # Parity matrix must have a 0 column, see Lemma 4.10, https://arxiv.org/abs/1312.6584.
    # We move it to be last.
    for i in range(2):
        if np.all(parity[:, i] == 0):
            parity[:, [i, 2]] = parity[:, [2, i]]
            break
    if np.array_equal(parity, np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])):
        # Leftmost syllable is T
        return ['T']
    elif np.array_equal(parity, np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])):
        # Leftmost syllable is HT
        return ['H', 'T']
    elif np.array_equal(parity, np.array([[1, 1, 0], [0, 0, 0], [1, 1, 0]])):
        # Leftmost syllable is SHT
        return ['S', 'H', 'T']
    else:
        raise ValueError(f'Unexpected parity matrix:\n{parity}')


def _matsumoto_amano_sequence(matrix: _su2_ct.SU2CliffordT) -> tuple[str, ...]:
    r"""Represents the Clifford+T operator in the Matsumoto-Amano normal form.

    Returns:
        a list of gates matching the regular expression $(T|\eps)(HT|SHT)^*C$,
        where C is a Clifford operator, itself represented as a list of H and S gates.
        The gates are returned in the order they need to be applied to generate the
        input matrix.

    Raises:
        ValueError if during the decomposition an invalid SU2CliffordT matrix is created.
    """
    gates = []
    while matrix.det() != _TWO:
        syllable = _matsumoto_amano_syllable(matrix)
        gates += syllable
        for gate in syllable:
            matrix = _su2_ct.GATE_MAP[gate].adjoint() @ matrix
        new = matrix.scale_down()
        if new is None or not new.is_valid():
            raise ValueError('Invalid SU2CliffordT matrix')
        matrix = new
    return clifford(matrix) + tuple(gates[::-1])


def to_sequence(matrix: _su2_ct.SU2CliffordT, fmt: str = 'xyz') -> tuple[str, ...]:
    r"""Returns a sequence of Clifford+T that produces the given matrix.

    Allowable format strings are
     - 'xyz' uses Tx, Ty, Tz gates.
     - 'xz' uses $Tx, Tz, Tx^\dagger, Tz^\dagger$
     - 't' uses only Tz gates, and returns the Matsumoto-Amano normal form.

    Args:
        matrix: The matrix to represent.
        fmt: A string from the set {'xz', 'xyz', 't'} representing the allowed T gates described above.

    Returns:
        A tuple of strings representing the gates.

    Raises:
        ValueError: If `fmt` is not supported.
    """
    if fmt == 'xyz':
        return _xyz_sequence(matrix)
    if fmt == 'xz':
        return cast(tuple[str, ...], _xz_sequence(matrix))
    if fmt == 't':
        return _matsumoto_amano_sequence(matrix)
    raise ValueError(f'{type=} is not supported')


_CIRQ_GATE_MAP: Mapping[str, cirq.Gate] = {
    "I": cirq.I,
    "H": cirq.H,
    "S": cirq.S,
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
    "Tx": cirq.rx(math.pi / 4),
    "Ty": cirq.ry(math.pi / 4),
    "Tz": cirq.rz(math.pi / 4),
    "Tx*": cirq.rx(-math.pi / 4),
    "Ty*": cirq.ry(-math.pi / 4),
    "Tz*": cirq.rz(-math.pi / 4),
    "T": cirq.rz(math.pi / 4),
}


def _to_quirk_name(name: str, allow_global_phase: bool = False) -> str:
    if allow_global_phase:
        if name == "I":
            return "1"
        if name in ("X", "Y", "Z", "H"):
            return "\"" + name + "\""
        if name == "S":
            return "\"Z^½\""
        if name == "S*":
            return "\"Z^-½\""
        if name.startswith("T"):
            if name == "T":
                return "\"Z^¼\""
            if name.endswith("*"):
                return "\"" + name[1].upper() + "^-¼" + "\""
            return "\"" + name[1].upper() + "^¼" + "\""
    else:
        if name == "I":
            return "1"
        if name == "H":
            return "\"H\""
        if name in ("X", "Y", "Z"):
            return '{"id":"R%sft","arg":"-pi"}' % (name.lower())
        if name == "S":
            return '{"id":"Rzft","arg":"pi/2"}'
        if name == "S*":
            return '{"id":"Rzft","arg":"-pi/2"}'
        if name.startswith("T"):
            if name == "T":
                return '{"id":"Rzft","arg":"pi/4"}'
            angle = ['pi/4', '-pi/4'][name.endswith('*')]
            return '{"id":"R%sft","arg":"%s"}' % (name[1].lower(), angle)
    raise ValueError(f"{name=} is not supported")


def to_cirq(
    matrix: _su2_ct.SU2CliffordT, fmt: str, q: Optional[cirq.Qid] = None
) -> tuple[cirq.Operation]:
    """Retruns a representation of the matrix as a sequence of Cirq operations.

    Args:
        matrix: The matrix to represent.
        fmt: The gates to use (see the documentation of to_sequence).
        q: The qubit to use.
    Returns:
        A tuple of Cirq operations.
    """
    q = q or cirq.q(0)
    return tuple(_CIRQ_GATE_MAP[g](q) for g in to_sequence(matrix, fmt))


def to_quirk(
    matrix: _su2_ct.SU2CliffordT, fmt: str, allow_global_phase: bool = False
) -> tuple[str, ...]:
    """Retruns a representation of the matrix as a sequence of quirk symbols.

    Args:
        matrix: The matrix to represent.
        fmt: The gates to use (see the documentation of to_sequence).
        allow_global_phase: whether the result can have a global phase or not.
            If this is set then each gate (except H) gets replaced by SU2 version:
                - S -> Rz(pi/2)
                - T -> Rz(pi/4)
                - X -> Rx(-pi)
                - Y -> Ry(-pi)
                - Z -> Rz(-pi)
            Then a prefix composed of SXSX that corrects the phase difference caused by H gates
            is added.
    Returns:
        A tuple quirk symbols.
    """
    sequence = to_sequence(matrix, fmt)
    phase_correction = tuple[str, ...]()
    if not allow_global_phase:
        phase = sum(g == 'H' for g in sequence) % 4
        phase_correction = ('"Z^½"', '"X"', '"Z^½"', '"X"') * phase
    return phase_correction + tuple(_to_quirk_name(name, allow_global_phase) for name in sequence)
