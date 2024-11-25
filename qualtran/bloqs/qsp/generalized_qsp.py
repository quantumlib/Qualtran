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
from collections import Counter
from functools import cached_property
from typing import Iterable, Iterator, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np
from attrs import field, frozen
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

from qualtran import (
    Bloq,
    bloq_example,
    BloqDocSpec,
    CtrlSpec,
    DecomposeTypeError,
    GateWithRegisters,
    QBit,
    Register,
    Signature,
)
from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
from qualtran.linalg.polynomial.qsp_testing import assert_is_qsp_polynomial
from qualtran.symbolics import (
    is_symbolic,
    is_zero,
    Shaped,
    slen,
    smax,
    smin,
    SymbolicFloat,
    SymbolicInt,
)

if TYPE_CHECKING:
    import cirq

    from qualtran.resource_counting import BloqCountDictT, SympySymbolAllocator


def qsp_complementary_polynomial(
    P: Union[NDArray[np.number], Sequence[complex]],
    *,
    verify: bool = False,
    verify_precision: float = 1e-7,
) -> Sequence[complex]:
    r"""Computes the Q polynomial given P

    Computes polynomial $Q$ of degree at-most that of $P$, satisfying

        $$ \abs{P(e^{i\theta})}^2 + \abs{Q(e^{i\theta})}^2 = 1 $$

    for every $\theta \in \mathbb{R}$.

    The exact method for computing $Q$ is described in the proof of Theorem 4.
    The method computes an auxillary polynomial R, whose roots are computed
    and re-interpolated to obtain the required polynomial Q.

    Args:
        P: Co-efficients of a complex polynomial.
        verify: sanity check the computed polynomial roots (defaults to False).
        verify_precision: precision to compare values while verifying

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 4.
    """
    d = len(P) - 1  # degree

    # R(z) = z^d (1 - P^*(z) P(z))
    # obtained on simplifying Eq. 34, Eq. 35 by substituting H, T.
    # The definition of $R$ is given in the text from Eq.34 - Eq. 35,
    # following the chain of definitions below:
    #
    #     $$
    #     T(\theta) = \abs{P(e^{i\theta}),
    #     H = I - T,
    #     H = e^{-id\theta} R(e^{i\theta})
    #     $$
    #
    # Here H and T are defined on reals, so the initial definition of R is only on the unit circle.
    # We analytically continue this definition to the entire complex plane by replacing $e^{i\theta}$ by $z$.
    R = Polynomial.basis(d) - Polynomial(P) * Polynomial(np.conj(P[::-1]))
    roots = R.roots()

    # R is self-inversive, so larger_roots and smaller_roots occur in conjugate pairs.
    units: list[complex] = []  # roots r s.t. \abs{r} = 1
    larger_roots: list[complex] = []  # roots r s.t. \abs{r} > 1
    smaller_roots: list[complex] = []  # roots r s.t. \abs{r} < 1

    for r in roots:
        if verify:
            assert not (np.abs(r) <= verify_precision), "zero root!"
        if np.allclose(np.abs(r), 1):
            units.append(r)
        elif np.abs(r) > 1:
            larger_roots.append(r)
        else:
            smaller_roots.append(r)

    # pair up roots in `units`, claimed in Eq. 40 and the explanation preceding it.
    # all unit roots must have even multiplicity.
    paired_units: list[complex] = []
    unpaired_units: list[complex] = []
    for z in units:
        matched_z = None
        for w in unpaired_units:
            if np.allclose(z, w, rtol=verify_precision):
                matched_z = w
                break

        if matched_z is not None:
            paired_units.append(z)
            unpaired_units.remove(matched_z)
        else:
            unpaired_units.append(z)

    if verify:
        assert len(unpaired_units) == 0

        # verify that the non-unit roots indeed occur in conjugate pairs.
        def assert_is_permutation(A, B):
            assert len(A) == len(B)
            A = list(A)
            unmatched = []
            for z in B:
                for w in A:
                    if np.allclose(z, w, rtol=verify_precision, atol=verify_precision):
                        A.remove(w)
                        break
                else:
                    unmatched.append(z)

            assert len(unmatched) == 0

        assert_is_permutation(smaller_roots, 1 / np.array(larger_roots).conj())

    # Q = G \hat{G}, where
    # - \hat{G}^2 is the monomials which are unit roots of R, which occur in pairs.
    # - G*(z) G(z) is the interpolation of the conjugate paired non-unit roots of R,
    #   described in Eq. 37 - Eq. 38

    # Leading co-efficient of R described in Eq. 37.
    # Note: In the paper, the polynomial is interpolated from larger_roots,
    #       but this is swapped in our implementation to reduce the error in Q.
    c = R.coef[-1]
    scaling_factor = np.sqrt(np.abs(c * np.prod(larger_roots)))

    Q = scaling_factor * Polynomial.fromroots(paired_units + smaller_roots)

    return Q.coef


def qsp_phase_factors(
    P: Union[NDArray[np.number], Sequence[complex]], Q: Union[NDArray[np.number], Sequence[complex]]
) -> Tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Computes the QSP signal rotations for a given pair of polynomials.

    The QSP transformation is described in Theorem 3, and the algorithm for computing
    co-efficients is described in Algorithm 1.

    Args:
        P: Co-efficients of a complex polynomial.
        Q: Co-efficients of a complex polynomial.

    Returns:
        A tuple (theta, phi, lambda).
        theta and phi have length degree(P) + 1.

    Raises:
        ValueError: when P and Q have different degrees.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
            Motlagh and Wiebe. (2023). Theorem 3; Algorithm 1.
    """
    if len(P) != len(Q):
        raise ValueError("Polynomials P and Q must have the same degree.")

    S = np.array([P, Q])
    n = S.shape[1]

    theta = np.zeros(n)
    phi = np.zeros(n)
    lambd = 0

    def safe_angle(x):
        return 0 if np.isclose(x, 0, atol=1e-10) else np.angle(x)

    for d in reversed(range(n)):
        assert S.shape == (2, d + 1)

        a, b = S[:, d]
        theta[d] = np.arctan2(np.abs(b), np.abs(a))
        # \phi_d = arg(a / b)
        phi[d] = 0 if np.isclose(np.abs(b), 0, atol=1e-10) else safe_angle(a * np.conj(b))

        if d == 0:
            lambd = safe_angle(b)
        else:
            S = SU2RotationGate(theta[d], phi[d], 0).rotation_matrix.conj().T @ S
            S = np.array([S[0][1 : d + 1], S[1][0:d]])

    return theta, phi, lambd


def _to_tuple(x: Union[Iterable[complex], Shaped]) -> Union[Tuple[complex, ...], Shaped]:
    """mypy-compatible attrs converter for GeneralizedQSP.P and Q"""
    if isinstance(x, Shaped):
        return x
    return tuple(x)


@frozen
class GeneralizedQSP(GateWithRegisters):
    r"""Applies a QSP polynomial $P$ to a unitary $U$ to obtain a block-encoding of $P(U)$.

    Given a unitary $U$ and a QSP polynomial $P$ (and its complementary polynomial $Q$),
    this gate implements the following unitary:

    $$
        \begin{bmatrix} P(U) & \cdot \\ Q(U) & \cdot \end{bmatrix}
    $$

    The polynomials $P$ and $Q$ should satisfy:

    $$
        \left|P(e^{i \theta})\right|^2 + \left|Q(e^{i \theta})\right|^2 = 1 ~~\text{for every}~ \theta \in \mathbb{R}
    $$

    The polynomial $P$ is said to be a QSP Polynomial if it satisfies:

    $$
        \left|P(e^{i \theta})\right|^2 \le 1 ~~\text{for every}~ \theta \in \mathbb{R}
    $$

    If only the QSP polynomial $P$ is known, one can simply call
    `GeneralizedQSP.from_qsp_polynomial(U, P)` which automatically computes $Q$.

    ### Using Laurent Polynomials
    To apply GQSP with the transformation given by $P'$

    $$
    P(z) = \sum_{n = -a}^b p_n z^n
    $$

    where $a, b \ge 0$, we can simply invoke GQSP with the standard polynomial $P'(z) = z^a P(z)$
    which has degree $a + b$, and pass `negative_power=a`.

    Given complementary QSP polynomials $P', Q'$ and `negative_power=a`,
    this gate implements the unitary transform:

    $$
        \begin{bmatrix} U^{-a} P'(U) & \cdot \\ U^{-a} Q'(U) & \cdot \end{bmatrix}
    $$


    The exact circuit implemented by this gate is described in Figure 2.

    Args:
        U: Unitary operation.
        P: Co-efficients of a complex QSP polynomial.
        Q: Co-efficients of a complex QSP polynomial.
        negative_power: value of $k$, which effectively applies $z^{-k} P(z)$. defaults to 0.
        precision: The error in the synthesized unitary. This is used to compute the required
                   precision for each single qubit SU2 rotation.

    References:
        [Generalized Quantum Signal Processing](https://arxiv.org/abs/2308.01501)
        Motlagh and Wiebe. (2023). Theorem 3; Figure 2; Theorem 6.
    """

    U: GateWithRegisters
    P: Union[Tuple[complex, ...], Shaped] = field(converter=_to_tuple)
    Q: Union[Tuple[complex, ...], Shaped] = field(converter=_to_tuple)
    negative_power: SymbolicInt = field(default=0, kw_only=True)
    precision: SymbolicFloat = field(default=1e-11, kw_only=True)

    @P.validator
    @Q.validator
    def _check_polynomial(self, attribute, value):
        degree = slen(value) - 1
        if not is_symbolic(degree) and degree <= 0:
            raise ValueError("GQSP Polynomial must have degree at least 1")

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register('signal', QBit()), *self.U.signature])

    @classmethod
    def from_qsp_polynomial(
        cls,
        U: GateWithRegisters,
        P: Union[NDArray[np.number], Sequence[complex], Shaped],
        *,
        negative_power: SymbolicInt = 0,
        precision: SymbolicFloat = 0,
        verify: bool = False,
        verify_precision=1e-7,
    ) -> 'GeneralizedQSP':
        if isinstance(P, Shaped) or is_symbolic(P):
            return GeneralizedQSP(U, P, P, negative_power=negative_power)

        if verify:
            assert_is_qsp_polynomial(P)
        Q = qsp_complementary_polynomial(P, verify=verify, verify_precision=verify_precision)
        return GeneralizedQSP(U, P, Q, negative_power=negative_power, precision=precision)

    @cached_property
    def _qsp_phases(self) -> Tuple[NDArray[np.floating], NDArray[np.floating], float]:
        if isinstance(self.P, Shaped) or isinstance(self.Q, Shaped):
            raise ValueError(
                'Cannot compute phases for symbolic GQSP polynomials {self.P=}, {self.Q=}'
            )
        return qsp_phase_factors(self.P, self.Q)

    @cached_property
    def _eps_per_rotation(self):
        """precision to synthesize each SU2 rotation."""
        return self.precision / (slen(self.P) + 1)

    @cached_property
    def signal_rotations(self) -> NDArray[SU2RotationGate]:  # type: ignore[type-var]
        thetas, phis, lambd = self._qsp_phases

        return np.array(
            [
                SU2RotationGate(theta, phi, lambd if i == 0 else 0, eps=self._eps_per_rotation)
                for i, (theta, phi) in enumerate(zip(thetas, phis))
            ]
        )

    def decompose_from_registers(
        self, *, context: 'cirq.DecompositionContext', signal, **quregs: NDArray['cirq.Qid']  # type: ignore[type-var]
    ) -> Iterator['cirq.OP_TREE']:
        if self.is_symbolic():
            raise DecomposeTypeError(f'Cannot decompose symbolic {self=}')

        (signal_qubit,) = signal

        num_inverse_applications = int(self.negative_power)

        yield self.signal_rotations[0].on(signal_qubit)
        for signal_rotation in self.signal_rotations[1:]:
            if num_inverse_applications > 0:
                # apply C-U^\dagger
                yield self.U.adjoint().on_registers(**quregs).controlled_by(signal_qubit)
                num_inverse_applications -= 1
            else:
                # apply C[0]-U
                yield self.U.on_registers(**quregs).controlled_by(signal_qubit, control_values=[0])
            yield signal_rotation.on(signal_qubit)

        for _ in range(num_inverse_applications):
            yield self.U.adjoint().on_registers(**quregs)

    def is_symbolic(self) -> bool:
        return is_symbolic(self.P, self.Q, self.negative_power)

    def build_call_graph(self, ssa: 'SympySymbolAllocator') -> 'BloqCountDictT':
        counts = Counter[Bloq]()

        degree = slen(self.P) - 1
        counts[SU2RotationGate.arbitrary(ssa)] += degree + 1
        counts[self.U.controlled(ctrl_spec=CtrlSpec(cvs=0))] += smax(
            0, degree - self.negative_power
        )
        counts[self.U.adjoint()] += smax(0, self.negative_power - degree)
        counts[self.U.adjoint().controlled()] += smin(degree, self.negative_power)

        return {bloq: count for bloq, count in counts.items() if not is_zero(count)}


@bloq_example
def _gqsp() -> GeneralizedQSP:
    from qualtran.bloqs.basic_gates import XPowGate

    gqsp = GeneralizedQSP.from_qsp_polynomial(XPowGate(), (0.5, 0.5))
    return gqsp


@bloq_example
def _gqsp_with_negative_power() -> GeneralizedQSP:
    from qualtran.bloqs.basic_gates import XPowGate

    gqsp_with_negative_power = GeneralizedQSP.from_qsp_polynomial(
        XPowGate(), (0.5, 0, 0.5), negative_power=1
    )
    return gqsp_with_negative_power


@bloq_example
def _gqsp_with_large_negative_power() -> GeneralizedQSP:
    from qualtran.bloqs.basic_gates import XPowGate

    gqsp_with_large_negative_power = GeneralizedQSP.from_qsp_polynomial(
        XPowGate(), (0.5, 0, 0.5), negative_power=5
    )
    return gqsp_with_large_negative_power


_Generalized_QSP_DOC = BloqDocSpec(
    bloq_cls=GeneralizedQSP,
    import_line='from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP',
    examples=[_gqsp, _gqsp_with_negative_power, _gqsp_with_large_negative_power],
)
