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

import functools
import itertools
from typing import cast, Mapping, Optional, Sequence, Union

import attrs
import numpy as np

import qualtran.rotation_synthesis._math_config as mc
from qualtran.rotation_synthesis.rings import _zsqrt2, _zw


@attrs.frozen(eq=False)
class SU2CliffordT:
    r"""A scaled $SU(2)$ unitary exactly synthesizable with Clifford+T gateset.

    An $SU(2)$ matrix that can be synthetized exactly with Clifford+T gates can be written as
    $$
        \frac{1}{\sqrt{2(2+\sqrt{2})^n}} \begin{bmatrix}
        u & -v^*\\
        v & u^*\\
        \end{bmatrix}
    $$
    Where $u, v \in \mathbb{Z}[e^{i \pi/4}]$ and $n$ is the needed number of $T$ gates with a
    determinant $\frac{1}{2(2+\sqrt{2})^n} (|u|^2 + |v|^2) = 1$.

    Instances of this class represent the matrix `[[u, -v^*], [v, u^*]]` with the scaling factor
    being implicit.

    Attributes:
        matrix: The scaled version of the $SU(2)$ unitary.
        gates: A tuple of strings representing the Clifford+T gates whose action gives this unitary
            The gates are given in circuit order and the property is present only if the object is
            constructed through multiplication.
    """

    matrix: np.ndarray = attrs.field(converter=np.asarray)
    gates: Optional[tuple[str, ...]] = None

    def __mul__(self, other):
        assert not isinstance(other, SU2CliffordT)
        return SU2CliffordT(self.matrix * other, self.gates)

    def __matmul__(self, other: "SU2CliffordT") -> "SU2CliffordT":
        res = self.matrix @ other.matrix
        for v in res.flat:
            assert v.is_divisible_by(_zw.SQRT_2)
        gates: Optional[tuple[str, ...]] = None
        if self.gates is not None and other.gates is not None:
            gates = other.gates + self.gates
        return SU2CliffordT([[v // _zw.SQRT_2 for v in r] for r in res], gates)

    def __rmul__(self, other):
        assert not isinstance(other, SU2CliffordT)
        return self * other

    def __neg__(self):
        return SU2CliffordT(-self.matrix)

    def __add__(self, other):
        return SU2CliffordT(self.matrix + other.matrix)

    def __sub__(self, other):
        return SU2CliffordT(self.matrix - other.matrix)

    def __hash__(self):
        return hash(tuple(map(tuple, self.matrix)))

    def __eq__(self, other):
        return np.all(self.matrix == other.matrix)

    def numpy(self, config: Optional[mc.MathConfig] = None) -> np.ndarray:
        """Returns the numpy representation of the unitary.
        Args:
            config: An optional MathConfig used to convert the matrix entries to complex
                numbers and normalize the result. If not given numpy methods are used.
        """
        if config is None:
            result = self.matrix.astype(complex)
            result = result / np.linalg.det(result) ** 0.5
            return result
        result = np.zeros((2, 2)) + 1j * config.zero
        sqrt_det = config.sqrt(self.det().value(config.sqrt2))
        for i in range(2):
            for j in range(2):
                result[i, j] = self.matrix[i, j].value(config.sqrt2)
        result = result / sqrt_det
        return result

    def adjoint(self) -> "SU2CliffordT":
        return SU2CliffordT(self.matrix.T.conj())

    def scale_down(self) -> Union["SU2CliffordT", None]:
        for v in self.matrix.flat:
            if not v.is_divisible_by(_zw.LAMBDA_KLIUCHNIKOV):
                return None
        return SU2CliffordT(
            [[v // _zw.LAMBDA_KLIUCHNIKOV for v in r] for r in self.matrix], self.gates
        )

    def det(self) -> _zsqrt2.ZSqrt2:
        a, b, c, d = self.matrix.reshape(-1)
        res = a * d - b * c
        real, imag, need_w = res.to_zsqrt2()
        assert not need_w
        assert imag == _zsqrt2.Zero
        return real

    @staticmethod
    def from_sequence(seq: Sequence[str]) -> "SU2CliffordT":
        """Creates an SU2CliffordT from a Clifford+T gate sequence."""
        u = ISqrt2
        for g in seq:
            u = _gate_from_name(g) @ u
        return u

    def parametric_form(
        self,
    ) -> tuple[_zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2]:
        real0, imag0, n0 = self.matrix[0, 0].to_zsqrt2()
        d = imag0 * _zsqrt2.SQRT_2 + n0
        real1, imag1, n1 = self.matrix[1, 0].to_zsqrt2()
        if n1:
            b = imag1 + (_zsqrt2.One - d).divide_by_sqrt2()
        else:
            b = imag1 - imag0

        c = -real1 - (_zsqrt2.ZSqrt2(n1, 0) + d).divide_by_sqrt2()
        a = (real0 - b - c + (_zsqrt2.ZSqrt2(n0, 0) - d).divide_by_sqrt2()).divide_by_sqrt2()
        return a, b, c, d

    @staticmethod
    def from_parametric_form(
        pf: tuple[_zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2]
    ) -> "SU2CliffordT":
        res = np.array([_zw.Zero] * 4).reshape((2, 2))
        for a, m in zip(pf, PARAMETRIC_FORM_BASES):
            res += m * _zw.ZW.from_pair(a, _zsqrt2.Zero, False)
        return SU2CliffordT(res)

    @functools.cached_property
    def _key(self) -> tuple[int, int, int, int]:
        pf = self.parametric_form()
        # the gate choice depends on the parametric form mod 2 + sqrt(2)
        # for elements of Z[sqrt(2)] this is equivalent ot taking the mod with sqrt(2)
        # which is the same as the parity of the integer part.
        return cast(tuple[int, int, int, int], tuple(v.a % 2 for v in pf))

    @classmethod
    def from_pair(
        cls: type["SU2CliffordT"], p: _zw.ZW, q: _zw.ZW, pick_phase: bool = False
    ) -> "SU2CliffordT":
        """Creates an SU2CliffordT instance from a pair of ZW rst.

        The matrix is constructed as [[p, -q.conj()], [q, p.conj()]].
        If pick_phase is True, it tries different phases for q to find a valid SU2CliffordT.

        Args:
            p: The top-left element of the matrix.
            q: The bottom-left element of the matrix.
            pick_phase: Whether to try different phases for q to ensure validity.

        Returns:
            An instance of SU2CliffordT.

        Raises:
            ValueError: If a valid SU(2) matrix can't be construct from the give pair.
        """
        if pick_phase:
            for exponent in range(8):
                phase = _zw.Omega**exponent
                nq = q * phase
        if pick_phase:
            for exponent in range(8):
                phase = _zw.Omega**exponent
                nq = q * phase
                res = SU2CliffordT([[p, -nq.conj()], [nq, p.conj()]])
                if res.is_valid():
                    return res
            raise ValueError(f"can't construct a valid SU(2) matrix from the given pair {p=} {q=}")
        else:
            res = cls([[p, -q.conj()], [q, p.conj()]])
            if not res.is_valid():
                raise ValueError(
                    f"can't construct a valid SU(2) matrix from the given pair {p=} {q=}"
                )
            return res

    def is_valid(self) -> bool:
        det = self.det()
        l_v = _zsqrt2.ZSqrt2(2, 1)
        two = _zsqrt2.ZSqrt2(2, 0)
        while det > two and det.is_divisible_by(l_v):
            det = det // l_v
        if det != two:
            return False

        _, _, n0 = self.matrix[0, 0].to_zsqrt2()
        _, _, n1 = self.matrix[1, 0].to_zsqrt2()
        return n0 == n1

    def rescale(self) -> 'SU2CliffordT':
        r"""Rescales the matrix such that its determinant is minimized.

        The determinant of the unitary can be written as $2\lambda^n$ where $\lambda=2+\sqrt{l}$
        and $n$ is the number of $T$ gates needed to synthesize the matrix. When all entries of
        the matrix are divisible by $\lambda$ then we can divide through by $\lambda$ to reduce $n$
        """
        u = self
        while u.det() > 2 * _zsqrt2.LAMBDA_KLIUCHNIKOV:
            if not all(a.is_divisible_by(_zw.LAMBDA_KLIUCHNIKOV) for a in u.matrix.flat):
                break
            new_u = SU2CliffordT(
                [[x // _zw.LAMBDA_KLIUCHNIKOV for x in row] for row in u.matrix], u.gates
            )
            if not new_u.is_valid():
                break
            u = new_u
        return u

    def num_t_gates(self) -> int:
        """Returns the number of T gates needed to synthesize the matrix."""
        det = self.det()
        x = _zsqrt2.ZSqrt2(2)
        n = 0
        while x < det:
            x = x * _zsqrt2.LAMBDA_KLIUCHNIKOV
            n += 1
        assert x == det
        return n

    def bloch_sphere_form(self) -> tuple[np.ndarray, int]:
        r"""Represents the unitary operator as a scaled element of SO(3).

        The entries of the returned matrix are in $\mathbb{Z}[\sqrt{2}]$,
        scaled by $\frac{1}{2\sqrt{2}(2+\sqrt{2})^n}$. The scaling factor
        is implicit. The additional $\sqrt{2}$ factor is needed to ensure
        the entries are in $\mathbb{Z}[\sqrt{2}]$.

        See Section 4 in https://arxiv.org/abs/1312.6584.

        Returns:
            the scaled SO(3) matrix and the value n.
        """
        u = self.matrix[0, 0]
        v = self.matrix[1, 0]
        return (
            np.array(
                [
                    [
                        (u**2 - v.conj() ** 2).real_zsqrt2(),
                        (u**2 - v**2).imag_zsqrt2(),
                        2 * (u * v.conj()).real_zsqrt2(),
                    ],
                    [
                        -(u**2 - v.conj() ** 2).imag_zsqrt2(),
                        (u**2 + v**2).real_zsqrt2(),
                        -2 * (u * v.conj()).imag_zsqrt2(),
                    ],
                    [
                        -2 * (u * v).real_zsqrt2(),
                        -2 * (u * v).imag_zsqrt2(),
                        (u * u.conj() - v * v.conj()).real_zsqrt2(),
                    ],
                ]
            ),
            self.num_t_gates(),
        )

    def bloch_form_parity(self) -> np.ndarray:
        """Returns the n-parity of the SO(3) Bloch sphere representation.

        See Definition 4.7, https://arxiv.org/abs/1312.6584 for the definition of k-parity.
        From Lemma 4.10, the least denominator exponent for the Bloch form equals the T-count,
        and so the n-parity is well-defined.
        """
        bf, n = self.bloch_sphere_form()
        bf = bf * _zsqrt2.SQRT_2**n
        scale_factor = _zsqrt2.ZSqrt2(2, 0) * _zsqrt2.SQRT_2 * _zsqrt2.LAMBDA_KLIUCHNIKOV**n
        if not all(x.is_divisible_by(scale_factor) for x in bf.flat):
            raise ValueError(
                "Not all entries of the \\sqrt{2}^n * SO(3) matrix are divisible "
                f"by 2\\sqrt{2}(2+\\sqrt{2})^n. The matrix is:\n{bf}"
            )
        scaled_bf = [[x // scale_factor for x in row] for row in bf]
        return np.array([[x.a % 2 for x in row] for row in scaled_bf])


def _gate_from_name(gate_name: str) -> SU2CliffordT:
    adjoint = gate_name.count('*') % 2 == 1
    if gate_name.endswith('*'):
        first_star_index = gate_name.index('*')
        suffix = gate_name[first_star_index:]
        gate_name = gate_name[:first_star_index]
        assert suffix.count('*') == len(suffix)
    gate = GATE_MAP[gate_name]
    if adjoint:
        gate = gate.adjoint()
    return gate


@functools.cache
def _key_map():
    Ts = {"Tx": Tx, "Ty": Ty, "Tz": Tz}
    ret = {}
    for vec in itertools.product(range(2), repeat=4):
        if np.all(np.array(vec) == 0):
            pf = (_zsqrt2.ZSqrt2(2, 0),) * 4
        else:
            pf = cast(
                tuple[_zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2, _zsqrt2.ZSqrt2],
                tuple(_zsqrt2.ZSqrt2(v, 0) for v in vec),
            )
        g = SU2CliffordT.from_parametric_form(pf)
        best = (g.det(), "")
        for name, t in Ts.items():
            ng = (t.adjoint() @ g).scale_down()
            if ng is None:
                continue
            best = min(best, (ng.det(), name))
        if best[0] >= g.det():
            continue
        ret[vec] = best[1]
    return ret


# H gate scaled by sqrt(2) to make its elements belong to Z[w] and 1j to make its determinant = 2
HSqrt2 = _zw.J * SU2CliffordT(np.array([[_zw.One, _zw.One], [_zw.One, -_zw.One]]), ("H",))

# S gate scaled by sqrt(2) to make its elements belong to Z[w] and w^* to make its determinant = 2
SSqrt2 = (
    _zw.SQRT_2
    * _zw.Omega.conj()
    * SU2CliffordT(np.array([[_zw.One, _zw.Zero], [_zw.Zero, _zw.J]]), ("S",))
)

# T gate scaled by sqrt(2) * (1 + w^*) to make its determinant = 2(2+sqrt(2))
# TSqrt2 is equal to Tz below
TSqrt2 = (
    _zw.SQRT_2
    * (1 + _zw.Omega.conj())
    * SU2CliffordT(np.array([[_zw.One, _zw.Zero], [_zw.Zero, _zw.Omega]]), ("T",))
)

# Paulis
ISqrt2: SU2CliffordT = _zw.SQRT_2 * SU2CliffordT(
    np.array([[_zw.One, _zw.Zero], [_zw.Zero, _zw.One]]), ()
)

ZSqrt2: SU2CliffordT = -SSqrt2 @ SSqrt2
XSqrt2: SU2CliffordT = HSqrt2 @ ZSqrt2 @ HSqrt2.adjoint()
YSqrt2: SU2CliffordT = ZSqrt2 @ XSqrt2

_X = np.array([[_zw.Zero, _zw.One], [_zw.One, _zw.Zero]])
_Y = np.array([[_zw.Zero, -_zw.J], [_zw.J, _zw.Zero]])
_Z = np.array([[_zw.One, _zw.Zero], [_zw.Zero, -_zw.One]])
_I = np.array([[_zw.One, _zw.Zero], [_zw.Zero, _zw.One]])

# Tx, Ty, Tz scaled by sqrt(2*(2+sqrt(2)))
Tx = SU2CliffordT(_I * _zw.SQRT_2 + _I - _X * _zw.J, ("Tx",))
Ty = SU2CliffordT(_I * _zw.SQRT_2 + _I - _Y * _zw.J, ("Ty",))
Tz = SU2CliffordT(_I * _zw.SQRT_2 + _I - _Z * _zw.J, ("Tz",))
Ts = [Tx, Ty, Tz]


GATE_MAP: Mapping[str, SU2CliffordT] = {
    "I": ISqrt2,
    "S": SSqrt2,
    "H": HSqrt2,
    "Tx": Tx,
    "Ty": Ty,
    "Tz": Tz,
    "X": XSqrt2,
    "Y": YSqrt2,
    "Z": ZSqrt2,
    "T": TSqrt2,
}

PARAMETRIC_FORM_BASES = [
    _I * _zw.SQRT_2,
    _I + _X * _zw.J,
    _I + _Y * _zw.J,
    np.array([[_zw.Omega, _zw.Omega], [-_zw.Omega.conj(), _zw.Omega.conj()]]),
]
