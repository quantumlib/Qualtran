from functools import cached_property
from typing import Dict

import attrs
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Signature, Soquet, SoquetT
from qualtran.bloqs.qubitization.prepare import BlackBoxPrepare
from qualtran.bloqs.qubitization.reflect import Reflect
from qualtran.bloqs.qubitization.select_bloq import BlackBoxSelect
from qualtran.drawing import directional_text_box, WireSymbol


@frozen
class Walk(Bloq):
    r"""Constructs a Szegedy Quantum Walk operator using `select` and `prepare`.

    $\def\select{\mathrm{SELECT}} \def\prepare{\mathrm{PREPARE}} \def\r{\rangle} \def\l{\langle}$
    Constructs the walk operator $W = R_L \cdot \select$, which is a product of
    reflection $R_L = (2|L \r\l L| - I)$ and $\select=\sum_l |l\r\l l|H_l$.
    $L$ is the state prepared by $\prepare|\vec0\r = |L\r$

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces. For an arbitrary eigenstate $|k\r$ of $H$ with eigenvalue $E_k$, $|\ell\r|k\r$
    and an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell\r|k\r$
    is in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2\arccos(E_k / \lambda)$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $\mathrm{spectrum}(H) = \lambda \cos(\arg(\mathrm{spectrum}(W)))$
    where $\arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $\select=\sum_l |l\r\l l|H_l$.
        prepare: The PREPARE lcu gate implementing $\prepare|\vec0>
            = \sum_l \sqrt{\frac{w_l}{\lambda}} |l\r = |\ell\r$
        power: Constructs $W^n$ by repeatedly decomposing into `power` copies of $W$. Defaults to 1.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    select: BlackBoxSelect
    prepare: BlackBoxPrepare
    power: int = 1

    def __attrs_post_init__(self):
        sb = self.select.selection_bitsize
        pb = self.prepare.bitsize
        if sb != pb:
            raise ValueError(
                f"Incompatible `select` and `prepare` bloqs: "
                f"The selection bitsize must match. Received {sb} and {pb}"
            )

    @cached_property
    def signature(self) -> Signature:
        return Signature.build(
            selection=self.select.selection_bitsize, system=self.select.system_bitsize
        )

    @cached_property
    def reflect(self):
        return Reflect(prepare=self.prepare)

    def build_composite_bloq(
        self, bb: 'BloqBuilder', selection: 'SoquetT', system: 'SoquetT'
    ) -> Dict[str, 'SoquetT']:
        for _ in range(self.power):
            selection, system = bb.add(self.select, selection=selection, system=system)
            (selection,) = bb.add(self.reflect, selection=selection)

        return {'selection': selection, 'system': system}

    def __pow__(self, power, modulo=None):
        assert modulo is None
        return attrs.evolve(self, power=self.power * power)

    def short_name(self) -> str:
        if self.power == 1:
            return 'W'
        return f'W^{self.power}'

    def wire_symbol(self, soq: 'Soquet') -> 'WireSymbol':
        if soq.reg.name == 'selection':
            return directional_text_box('sel', soq.reg.side)
        if soq.reg.name == 'system':
            return directional_text_box('sys', soq.reg.side)
