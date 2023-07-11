from functools import cached_property
from typing import Dict

import cirq
import numpy as np
from attrs import frozen
from numpy.typing import NDArray

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.qubitization.walk import Walk
from qualtran.bloqs.state_prep_channel import StatePrepChannel
from qualtran.cirq_interop import CirqGateAsBloq


def get_resource_state(m: int) -> NDArray:
    r"""Returns a state vector representing the resource state on m qubits from Eq.17 of Ref-1.

    Returns a numpy array of size 2^{m} representing the state vector corresponding to the state
    $$
        \sqrt{\frac{2}{2^m + 1}} \sum_{n=0}^{2^{m}-1} \sin{\frac{\pi(n + 1)}{2^{m}+1}}\ket{n}
    $$

    Args:
        m: Number of qubits to prepare the resource state on.

    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Eq. 17
    """
    den = 1 + 2**m
    norm = np.sqrt(2 / den)
    return norm * np.sin(np.pi * (1 + np.arange(2**m)) / den)


@frozen
class PhaseEstimation(Bloq):
    """Heisenberg limited phase estimation circuit for learning eigenphase of `walk`.

     Heisenberg limited phase estimation circuit
    for learning eigenphases of the `walk` operator with `m` bits of accuracy. The
    circuit is implemented as given in Fig.2 of Ref-1.

    Args:
        walk: Qubitization walk operator.
        m: Number of bits of accuracy for phase estimation.

    Ref:
        1) [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
            (https://arxiv.org/abs/1805.03662)
            Fig. 2
    """

    walk: Walk
    m: int

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build(
            phase=self.m,
            selection=self.walk.select.selection_bitsize,
            system=self.walk.select.system_bitsize,
        )

    def build_composite_bloq(
        self, bb: 'BloqBuilder', phase: Soquet, selection: Soquet, system: Soquet
    ) -> Dict[str, 'SoquetT']:
        reflect = self.walk.reflect

        creflect = reflect.controlled()  # TODO: cv = 0
        walk = self.walk
        cwalk = walk.controlled()

        (phase,) = bb.add(
            StatePrepChannel(bitsize=self.m, target_state=tuple(get_resource_state(self.m))),
            x=phase,
        )
        phase_bits = bb.split(phase)

        phase_bits[0], selection, system = bb.add(
            cwalk, ctrl=phase_bits[0], selection=selection, system=system
        )
        for i in range(1, self.m):
            phase_bits[i], selection = bb.add(creflect, ctrl=phase_bits[i], selection=selection)
            selection, system = bb.add(walk, selection=selection, system=system)
            walk = walk**2
            phase_bits[i], selection = bb.add(creflect, ctrl=phase_bits[i], selection=selection)

        # TODO: inverse
        (phase_bits,) = bb.add(
            CirqGateAsBloq(cirq.QuantumFourierTransformGate(self.m)), qubits=phase_bits
        )
        return dict(phase=bb.join(phase_bits), selection=selection, system=system)
