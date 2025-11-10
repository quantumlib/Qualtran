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
from typing import cast, Mapping, Optional

import cirq

import qualtran.rotation_synthesis.matrix.generation as ctg
import qualtran.rotation_synthesis.matrix.su2_ct as su2_ct
import qualtran.rotation_synthesis.rings.zsqrt2 as zsqrt2

_TWO = zsqrt2.ZSqrt2(2)


def clifford(matrix: su2_ct.SU2CliffordT) -> tuple[str, ...]:
    assert matrix.det() == 2
    cliffords = ctg.generate_cliffords()
    if matrix in cliffords:
        return cliffords[matrix]
    return ("Z", "X", "Z", "X") + cliffords[-matrix]


def _xyz_sequence(matrix: su2_ct.SU2CliffordT) -> tuple[str, ...]:
    seq = []
    while matrix.det() > _TWO:
        t = su2_ct._key_map()[matrix._key]
        nxt = (su2_ct.GATE_MAP[t].adjoint() @ matrix).scale_down()
        assert nxt is not None
        assert nxt.det() < matrix.det()
        seq.append(t)
        matrix = nxt
    return clifford(matrix) + tuple(seq[::-1])


_T_list = [
    ('Tz', su2_ct.Tz),
    ('Tx', su2_ct.Tx),
    ('Tz*', su2_ct.Tz.adjoint()),
    ('Tx*', su2_ct.Tx.adjoint()),
]


def _xz_sequence(matrix: su2_ct.SU2CliffordT, use_hs: bool = True) -> Optional[tuple[str, ...]]:
    if matrix.det() == 2:
        return clifford(matrix)
    cliffords = [su2_ct.ISqrt2]
    if use_hs:
        cliffords.append(su2_ct.HSqrt2)
        cliffords.append(su2_ct.HSqrt2 @ su2_ct.SSqrt2)
    candidates = []
    for name, t in _T_list:
        for c in cliffords:
            new = c.adjoint() @ matrix
            new = t.adjoint() @ new
            new = new.scale_down()
            if new is None or not new.is_valid():
                continue
            seq = _xz_sequence(new, False)
            if seq is None:
                continue
            gates = cast(tuple[str, ...], c.gates)
            candidates.append(seq + (name,) + gates)
    if not candidates:
        return None
    return min(candidates, key=len)


def to_sequence(matrix: su2_ct.SU2CliffordT, fmt: str = 'xyz') -> tuple[str, ...]:
    r"""Returns a sequence of Clifford+T that produces the given matrix.

    Args:
        matrix: The matrix to represent.
        fmt: A string from the set {'xz', 'xyz'} representing the allowed T gates where
            - 'xyz' uses Tx, Ty, Tz gates.
            - 'xz' uses $Tx, Tz, Tx^\dagger, Tz^\dagger$
    Returns:
        A tuple of strings representing the gates.
    Raises:
        ValueError: If `fmt` is not supported.
    """
    if fmt == 'xyz':
        return _xyz_sequence(matrix)
    if fmt == 'xz':
        return cast(tuple[str, ...], _xz_sequence(matrix))
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
}


def _to_quirk_name(name: str) -> str:
    if name == "I":
        return "1"
    if name in ("X", "Y", "Z", "H"):
        return "\"" + name + "\""
    if name == "S":
        return "\"Z^½\""
    if name == "S*":
        return "\"Z^-½\""
    if name.startswith("T"):
        if name.endswith("*"):
            return "\"" + name[1].upper() + "^-¼" + "\""
        return "\"" + name[1].upper() + "^¼" + "\""
    raise ValueError(f"{name=} is not supported")


def to_cirq(
    matrix: su2_ct.SU2CliffordT, fmt: str, q: Optional[cirq.Qid] = None
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


def to_quirk(matrix: su2_ct.SU2CliffordT, fmt: str) -> tuple[str, ...]:
    """Retruns a representation of the matrix as a sequence of quirk symbols.

    Args:
        matrix: The matrix to represent.
        fmt: The gates to use (see the documentation of to_sequence).
    Returns:
        A tuple quirk symbols.
    """
    return tuple(_to_quirk_name(name) for name in to_sequence(matrix, fmt))
