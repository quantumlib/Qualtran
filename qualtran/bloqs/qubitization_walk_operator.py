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
from typing import Iterator, Optional, Tuple

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, CtrlSpec, Register, Signature
from qualtran._infra.gate_with_registers import SpecializedSingleQubitControlledGate, total_bits
from qualtran.bloqs.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.select_and_prepare import PrepareOracle, SelectOracle
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.symbolics import SymbolicFloat


@attrs.frozen(cache_hash=True)
class QubitizationWalkOperator(SpecializedSingleQubitControlledGate):
    r"""Constructs a Szegedy Quantum Walk operator using LCU oracles SELECT and PREPARE.

    For a Hamiltonian $H = \sum_l w_l H_l$ (s.t. $w_l > 0$ and $H_l$ are unitaries),
    Constructs a Szegedy quantum walk operator $W = R_{L} . SELECT$, which is a product of
    two reflections $R_{L} = (2|L><L| - I)$ and $SELECT=\sum_{l}|l><l|H_{l}$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
        vector spaces. For an arbitrary eigenstate $|k>$ of $H$ with eigenvalue $E_k$, $|\ell>|k>$ and
    an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell>|k>$ is
    in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2arccos(E_{k} / \lambda)$ s.t. $W = e^{i arccos(E_k / \lambda) Y}$,
    where $\lambda = \sum_l w_l$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $spectrum(H) = \lambda cos(\arg(\mathrm{spectrum}(W)))$ where $\arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $\mathrm{SELECT}=\sum_{l}|l\rangle\langle l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $\mathrm{PREPARE}|00...00\rangle = \sum_{l=0}^{L - 1}\sqrt{\frac{w_{l}}{\lambda}}
            |l\rangle = |\ell\rangle$
        control_val: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    select: SelectOracle
    prepare: PrepareOracle
    control_val: Optional[int] = None
    uncompute: bool = False

    def __attrs_post_init__(self):
        assert self.select.control_registers == self.reflect.control_registers

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.select.control_registers

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return self.prepare.selection_registers

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return self.select.target_registers

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(self.prepare, control_val=self.control_val, global_phase=-1)

    @cached_property
    def sum_of_lcu_coefficients(self) -> Optional[SymbolicFloat]:
        r"""value of $\lambda$, i.e. sum of absolute values of coefficients $w_l$."""
        return self.prepare.l1_norm_of_coeffs

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        select_reg = {reg.name: quregs[reg.name] for reg in self.select.signature}

        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
        if self.uncompute:
            yield self.reflect.adjoint().on_registers(**reflect_reg)
            yield self.select.adjoint().on_registers(**select_reg)

        else:
            yield self.select.on_registers(**select_reg)
            yield self.reflect.on_registers(**reflect_reg)

    def get_single_qubit_controlled_bloq(self, control_val: int) -> 'QubitizationWalkOperator':
        assert self.control_val is None

        c_select = self.select.controlled(ctrl_spec=CtrlSpec(cvs=control_val))
        if not isinstance(c_select, SelectOracle):
            raise TypeError(
                f"controlled version of {self.select} = {c_select} must also be a SelectOracle"
            )
        return attrs.evolve(self, select=c_select, control_val=control_val)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self.control_val else '@(0)'] * total_bits(self.control_registers)
        wire_symbols += ['W'] * (total_bits(self.signature) - total_bits(self.control_registers))
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def adjoint(self) -> 'QubitizationWalkOperator':
        return attrs.evolve(self, uncompute=not self.uncompute)


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _walk_op() -> QubitizationWalkOperator:
    from qualtran.bloqs.qubitization_walk_operator_test import get_walk_operator_for_1d_ising_model

    walk_op = get_walk_operator_for_1d_ising_model(4, 2e-1)
    return walk_op


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _thc_walk_op() -> QubitizationWalkOperator:
    from openfermion.resource_estimates.utils import QI

    from qualtran.bloqs.chemistry.thc.walk_operator import get_walk_operator_for_thc_ham

    # Li et al parameters from openfermion.resource_estimates.thc.compute_cost_thc_test
    num_spinorb = 152
    num_bits_state_prep = 10
    num_bits_rot = 20
    thc_dim = 450
    num_spat = num_spinorb // 2
    tpq = np.random.normal(size=(num_spat, num_spat))
    tpq = 0.5 * (tpq + tpq) / 2
    zeta = np.random.normal(size=(thc_dim, thc_dim))
    zeta = 0.5 * (zeta + zeta) / 2
    qroam_blocking_factor = np.power(2, QI(thc_dim + num_spat)[0])
    thc_walk_op = get_walk_operator_for_thc_ham(
        tpq,
        zeta,
        num_bits_state_prep=num_bits_state_prep,
        num_bits_theta=num_bits_rot,
        kr1=qroam_blocking_factor,
        kr2=qroam_blocking_factor,
    )
    return thc_walk_op


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _walk_op_chem_sparse() -> QubitizationWalkOperator:
    from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.sparse.walk_operator import get_walk_operator_for_sparse_chem_ham

    num_spin_orb = 8
    num_bits_rot_aa = 8
    num_bits_state_prep = 12
    tpq, eris = build_random_test_integrals(num_spin_orb // 2)
    walk_op_chem_sparse = get_walk_operator_for_sparse_chem_ham(
        tpq, eris, num_bits_rot_aa=num_bits_rot_aa, num_bits_state_prep=num_bits_state_prep
    )
    return walk_op_chem_sparse


_QUBITIZATION_WALK_DOC = BloqDocSpec(
    bloq_cls=QubitizationWalkOperator,
    import_line='from qualtran.bloqs.qubitization_walk_operator import QubitizationWalkOperator',
    examples=(_walk_op, _thc_walk_op, _walk_op_chem_sparse),
)
