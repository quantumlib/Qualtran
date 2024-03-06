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

from attrs import frozen

from qualtran import (
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.arithmetic import SumOfSquares
from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation


@frozen
class KineticEnergy(Bloq):
    r"""Bloq for the Kinetic energy unitary defined in the reference.

    Args:
        num_elec: The number of electrons.
        num_grid: The number of grid points in each of the x, y and z
            directions. In total, for a cubic grid, there are $N = \mathrm{num\_grid}^3$
            grid points. The number of bits required (in each spatial dimension)
            is thus log N + 1, where the + 1 is for the sign bit.

    Registers:
        system: The system register of size eta * 3 * nb

    References:
        [Faster quantum chemistry simulation on fault-tolerant quantum
            computers](https://iopscience.iop.org/article/10.1088/1367-2630/14/11/115023/meta)
    """

    num_elec: int
    num_grid: int

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register(
                    'system', QAny(((self.num_grid - 1).bit_length() + 1)), shape=(self.num_elec, 3)
                )
            ]
        )

    def short_name(self) -> str:
        return 'U_T(dt)'

    def build_composite_bloq(self, bb: BloqBuilder, *, system: SoquetT) -> Dict[str, SoquetT]:
        bitsize = (self.num_grid - 1).bit_length() + 1
        for i in range(self.num_elec):
            system[i], sos = bb.add(SumOfSquares(bitsize=bitsize, k=3), input=system[i])
            sos = bb.add(QuantumVariableRotation(phi_bitsize=(2 * bitsize + 2)), phi=sos)
            bb.free(sos)
        return {'system': system}


@bloq_example
def _kinetic_energy() -> KineticEnergy:
    nelec = 12
    ngrid_x = 2 * 8 + 1
    kinetic_energy = KineticEnergy(nelec, ngrid_x)
    return kinetic_energy


_KINETIC_ENERGY = BloqDocSpec(
    bloq_cls=KineticEnergy,
    import_line='from qualtran.bloqs.chemistry.trotter.grid_ham.kinetic import KineticEnergy',
    examples=(_kinetic_energy,),
)
