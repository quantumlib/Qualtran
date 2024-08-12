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
"""Bloq for building a Trotterized unitary"""

from functools import cached_property
from typing import Dict, Sequence

import attrs

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, SoquetT
from qualtran.symbolics import SymbolicFloat


@attrs.frozen
class TrotterizedUnitary(Bloq):
    r"""Implement arbitrary trotterized unitary given any Trotter splitting of the Hamiltonian.

    Given an arbitrary splitting of the Hamiltonian into $\Gamma$ terms

    $$
        H = \sum_{\gamma=1}^\Gamma H_\gamma
    $$

    then the unitary $e^{-i t H}$ can be approximately implemented via a $p$-th order product
    formula

    $$
        S_p(t) = \prod_{v=1}^{\Upsilon}\prod_{\gamma=1}^\Gamma e^{-it a_{v,\gamma} H_{\pi_v(\gamma)}}
    $$

    where $\Upsilon$ is the number of `stages`, $a_{v, \gamma}$ are real numbers
    and $\pi_v(\gamma)$ is a permutation of the Hamiltonian term labels.

    In practice, to construct the unitary we adopt the convention from the second reference
    which expands the product above and merges neighbouring unitaries where
    possible.
    In particular, the trotterized unitary can be specified by

    $$
        S_p(t) = \prod_{k}^M e^{-it c_k H_{l_k}}
    $$

    where the coefficients $c_k$ are real numbers and $l_k$ is an integer
    indexing which term of the Hamiltonian to apply.

    For example, the second order Suzuki splitting would have indicies $(l)$ = (0, 1, 0)
    and coeffs = $(c)$ = (0.5, 1, 0.5), which would build

    $$
        e^{-i \frac{t}{2} H_0} e^{-i t H_1} e^{-i \frac{t}{2} H_0}
    $$

    Args:
        bloqs: A tuple of bloqs of length $\Gamma$ which implement the unitaries for
            each term in the Hamiltonian. Each bloq should be a frozen attrs
            dataclass, have an `angle` parameter. All bloqs should have the same signature.
        indices: A tuple of integers which specifies which bloq to apply when
            forming the unitary as a product of unitaries.
        coeffs: The coefficients $a$ which appear in the expression for the unitary.
        timestep: The timestep $t$.

    Registers:
        system: The system register to which to apply the unitary.

    References:
        [Theory of Trotter Error with Commutator Scaling](
            https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020) Eq. 12 page 7.

        [Trotter error with commutator scaling for the Fermi-Hubbard model](
            https://arxiv.org/abs/2306.10603) see github repo for software to produce splittings.
    """

    bloqs: Sequence[Bloq]
    indices: Sequence[int]
    coeffs: Sequence[SymbolicFloat]
    timestep: SymbolicFloat

    def __attrs_post_init__(self):
        ref_sig = self.bloqs[0].signature
        for bloq in self.bloqs:
            if bloq.signature != ref_sig:
                raise ValueError(
                    f"Bloqs must have the same signature. Got {ref_sig} and {bloq.signature}"
                )
            if not attrs.has(bloq.__class__):
                raise ValueError("Bloq must be an attrs dataclass.")
            if attrs.fields_dict(bloq.__class__).get('angle') is None:
                raise ValueError("Bloq must have a parameter named 'angle'.")

    @cached_property
    def signature(self) -> Signature:
        return self.bloqs[0].signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: SoquetT) -> Dict[str, 'SoquetT']:
        for i, a in zip(self.indices, self.coeffs):
            # Bloqs passed in are supposed to be attrs dataclasses per docs
            # It would be nice to somehow specify that self.bloqs are both bloqs and AttrsInstance
            soqs |= bb.add_d(attrs.evolve(self.bloqs[i], angle=2 * a * self.timestep), **soqs)  # type: ignore[misc]
        return soqs


@bloq_example
def _trott_unitary() -> TrotterizedUnitary:
    from qualtran.bloqs.chemistry.trotter.ising import IsingXUnitary, IsingZZUnitary

    nsites = 3
    j_zz = 2
    gamma_x = 0.1
    dt = 0.01
    indices = (0, 1, 0)
    coeffs = (0.5 * gamma_x, j_zz, 0.5 * gamma_x)
    # The angles for the Trotter bloqs will be overwritten, so these are placeholder values.
    zz_bloq = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz)
    x_bloq = IsingXUnitary(nsites=nsites, angle=0.5 * 2 * dt * gamma_x)
    trott_unitary = TrotterizedUnitary(
        bloqs=(x_bloq, zz_bloq), indices=indices, coeffs=coeffs, timestep=dt
    )
    return trott_unitary


_TROTT_UNITARY_DOC = BloqDocSpec(
    bloq_cls=TrotterizedUnitary,
    import_line=(
        'from qualtran.bloqs.chemistry.trotter.trotterized_unitary import TrotterizedUnitary'
    ),
    examples=(_trott_unitary,),
)
