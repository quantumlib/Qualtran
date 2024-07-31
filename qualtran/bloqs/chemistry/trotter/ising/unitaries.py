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
from functools import cached_property
from typing import Dict

import attrs

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, Signature, Soquet
from qualtran.bloqs.basic_gates import CNOT, Rx, Rz


@attrs.frozen
class IsingXUnitary(Bloq):
    r"""Implents the unitary $e^{-i \alpha H_X}$.

    Args:
        nsites: The number of lattice sites.
        angle: The angle of the rotation. $\alpha$ in the docstring.
        eps: The tolerance for the rotation.

    Registers:
        system: The system register to apply the unitary to.
    """
    nsites: int
    angle: float
    eps: float = 1e-10

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(system=self.nsites)

    def pretty_name(self) -> str:
        return 'U_X'

    def build_composite_bloq(self, bb: 'BloqBuilder', system: 'Soquet') -> Dict[str, 'Soquet']:
        system = bb.split(system)
        for iq in range(self.nsites):
            system[iq] = bb.add(Rx(self.angle), q=system[iq])
        return {'system': bb.join(system)}


@attrs.frozen
class IsingZZUnitary(Bloq):
    r"""Implents the unitary $e^{-i \alpha H_{ZZ}}$.

    Args:
        nsites: The number of lattice sites.
        angle: The angle of the rotation. $\alpha$ in the docstring.
        eps: The tolerance for the rotation.

    Registers:
        system: The system register to apply the unitary to.
    """
    nsites: int
    angle: float
    eps: float = 1e-10

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(system=self.nsites)

    def pretty_name(self) -> str:
        return 'U_ZZ'

    def build_composite_bloq(self, bb: 'BloqBuilder', system: 'Soquet') -> Dict[str, 'Soquet']:
        system = bb.split(system)
        for iq_a in range(self.nsites):
            iq_b = (iq_a + 1) % self.nsites
            system[iq_a], system[iq_b] = bb.add(CNOT(), ctrl=system[iq_a], target=system[iq_b])
            system[iq_b] = bb.add(Rz(self.angle, self.eps), q=system[iq_b])
            system[iq_a], system[iq_b] = bb.add(CNOT(), ctrl=system[iq_a], target=system[iq_b])
        return {'system': bb.join(system)}


@bloq_example
def _ising_zz() -> IsingZZUnitary:
    nsites = 3
    j_zz = 2
    dt = 0.01
    ising_zz = IsingZZUnitary(nsites=nsites, angle=2 * dt * j_zz)
    return ising_zz


@bloq_example
def _ising_x() -> IsingXUnitary:
    nsites = 3
    j_zz = 2
    dt = 0.01
    ising_x = IsingXUnitary(nsites=nsites, angle=2 * dt * j_zz)
    return ising_x


_ISING_ZZ_UNITARY_DOC = BloqDocSpec(
    bloq_cls=IsingZZUnitary,
    import_line=('from qualtran.bloqs.chemistry.trotter.ising.unitaries import IsingZZUnitary'),
    examples=(_ising_zz,),
)

_ISING_X_UNITARY_DOC = BloqDocSpec(bloq_cls=IsingXUnitary, examples=(_ising_x,))
