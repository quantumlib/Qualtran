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

r"""Bloqs for constructing quantum walks from Select and Prepare operators.

The spectrum of a quantum Hamiltonian can be encoded in the spectrum of a quantum "walk"
operator. The Prepare and Select subroutines are carefully designed so that the Hamiltonian
$H$ is encoded as a projection of Select onto the state prepared by Prepare:

$$
\mathrm{PREPARE}|0\rangle = |\mathcal{L}\rangle \\
(\langle \mathcal{L} | \otimes \mathbb{1}) \mathrm{SELECT} (|\mathcal{L} \rangle \otimes \mathbb{1}) = H / \lambda
$$.
We first document the SelectOracle and PrepareOracle abstract base bloqs, and then show
how they can be combined in `QubitizationWalkOperator`.
"""

from functools import cached_property
from typing import Iterator, Optional, Tuple, Union

import attrs
import cirq
import numpy as np
from numpy.typing import NDArray

from qualtran import bloq_example, BloqDocSpec, CtrlSpec, Register, Signature
from qualtran._infra.gate_with_registers import GateWithRegisters, total_bits
from qualtran._infra.single_qubit_controlled import SpecializedSingleQubitControlledExtension
from qualtran.bloqs.block_encoding.lcu_block_encoding import (
    BlackBoxPrepare,
    LCUBlockEncoding,
    SelectBlockEncoding,
)
from qualtran.bloqs.reflections.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
from qualtran.resource_counting.generalizers import (
    cirq_to_bloqs,
    ignore_cliffords,
    ignore_split_join,
)
from qualtran.symbolics import SymbolicFloat


@attrs.frozen(cache_hash=True)
class QubitizationWalkOperator(GateWithRegisters, SpecializedSingleQubitControlledExtension):  # type: ignore[misc]
    r"""Construct a Szegedy Quantum Walk operator using LCU oracles SELECT and PREPARE.

    For a Hamiltonian $H = \sum_l w_l H_l$ (where coefficients $w_l > 0$ and $H_l$ are unitaries),
    This bloq constructs a Szegedy quantum walk operator $W = R_{L} \cdot \mathrm{SELECT}$,
    which is a product of two reflections:
     - $R_L = (2|L\rangle\langle L| - I)$ and
     - $\mathrm{SELECT}=\sum_l |l\rangle\langle l|H_l$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces giving it the name "qubitization".
    For an arbitrary eigenstate $|k\rangle$ of $H$ with eigenvalue $E_k$,
    the two-dimensional space is spanned by $|L\rangle|k\rangle$ and
    an orthogonal state $\phi_k$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2\arccos(E_k / \lambda)$ where $\lambda = \sum_l w_l$. That is,
    $W = e^{i \arccos(E_k / \lambda) Y}$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$,
    specifically $\mathrm{spectrum}(H) = \lambda \cos(\arg(\mathrm{spectrum}(W)))$
    where $\arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $\mathrm{SELECT}=\sum_{l}|l\rangle\langle l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $\mathrm{PREPARE}|0\dots 0\rangle = \sum_l \sqrt{\frac{w_{l}}{\lambda}}
            |l\rangle = |L\rangle$
        control_val: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Figure 1.
    """

    block_encoding: Union[SelectBlockEncoding, LCUBlockEncoding]
    control_val: Optional[int] = None
    uncompute: bool = False

    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return self.block_encoding.control_registers

    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        regs = tuple(
            set(self.block_encoding.selection_registers + self.reflect.selection_registers)
        )
        names = [r.name for r in regs]
        if len(names) != len(set(names)):
            msg = ''.join(f"{reg.name=}: {reg.dtype=} {reg.shape=}\n" for reg in regs)
            raise ValueError(
                "Found duplicate registers when creating union of block encoding selection "
                f"registers and reflection selection registers. Possible dtype mismatch. \n {msg}"
            )
        return regs

    @cached_property
    def target_registers(self) -> Tuple[Register, ...]:
        return self.block_encoding.target_registers

    @cached_property
    def junk_registers(self) -> Tuple[Register, ...]:
        return self.block_encoding.junk_registers

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                *self.control_registers,
                *self.selection_registers,
                *self.target_registers,
                *self.junk_registers,
            ]
        )

    @cached_property
    def reflect(self) -> ReflectionUsingPrepare:
        return ReflectionUsingPrepare(
            self.block_encoding.signal_state, control_val=self.control_val, global_phase=-1
        )

    @cached_property
    def sum_of_lcu_coefficients(self) -> SymbolicFloat:
        r"""value of $\lambda$, i.e. sum of absolute values of coefficients $w_l$."""
        return self.block_encoding.alpha

    def decompose_from_registers(
        self,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> Iterator[cirq.OP_TREE]:
        select_reg = {reg.name: quregs[reg.name] for reg in self.block_encoding.signature}

        reflect_reg = {reg.name: quregs[reg.name] for reg in self.reflect.signature}
        if self.uncompute:
            yield self.reflect.adjoint().on_registers(**reflect_reg)
            yield self.block_encoding.adjoint().on_registers(**select_reg)
        else:
            yield self.block_encoding.on_registers(**select_reg)
            yield self.reflect.on_registers(**reflect_reg)

    def get_single_qubit_controlled_bloq(self, control_val: int) -> 'QubitizationWalkOperator':
        assert self.control_val is None
        c_block = self.block_encoding.controlled(ctrl_spec=CtrlSpec(cvs=control_val))
        if not isinstance(c_block, (SelectBlockEncoding, LCUBlockEncoding)):
            raise TypeError(
                f"controlled version of {self.block_encoding} = {c_block} must also be a SelectOracle"
            )
        return attrs.evolve(self, block_encoding=c_block, control_val=control_val)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        wire_symbols = ['@' if self.control_val else '@(0)'] * total_bits(self.control_registers)
        wire_symbols += ['W'] * (total_bits(self.signature) - total_bits(self.control_registers))
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def adjoint(self) -> 'QubitizationWalkOperator':
        return attrs.evolve(self, uncompute=not self.uncompute)

    @cached_property
    def prepare(self) -> Union[PrepareOracle, BlackBoxPrepare]:
        """Get the Prepare bloq if appropriate from the block encoding."""
        # TODO: This should use self.signal_state
        # https://github.com/quantumlib/Qualtran/issues/1266
        if hasattr(self.block_encoding, 'prepare'):
            return self.block_encoding.prepare
        raise ValueError(
            f"Prepare bloq not implemented or not appropriate for {self.block_encoding}."
        )


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _walk_op() -> QubitizationWalkOperator:
    from qualtran.bloqs.chemistry.ising.walk_operator import get_walk_operator_for_1d_ising_model

    walk_op, _ = get_walk_operator_for_1d_ising_model(4, 2e-1)
    return walk_op


@bloq_example(generalizer=[cirq_to_bloqs, ignore_split_join, ignore_cliffords])
def _thc_walk_op() -> QubitizationWalkOperator:
    from openfermion.resource_estimates.utils import QI

    from qualtran.bloqs.chemistry.thc.prepare_test import build_random_test_integrals
    from qualtran.bloqs.chemistry.thc.walk_operator import get_walk_operator_for_thc_ham

    # Li et al parameters from openfermion.resource_estimates.thc.compute_cost_thc_test
    num_spinorb = 152
    num_bits_state_prep = 10
    num_bits_rot = 20
    thc_dim = 450
    num_spat = num_spinorb // 2
    t_l, eta, zeta = build_random_test_integrals(thc_dim, num_spinorb // 2, seed=7)
    qroam_blocking_factor = np.power(2, QI(thc_dim + num_spat)[0])
    thc_walk_op = get_walk_operator_for_thc_ham(
        t_l,
        eta,
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
    bloq_cls=QubitizationWalkOperator, examples=(_walk_op, _thc_walk_op, _walk_op_chem_sparse)
)
