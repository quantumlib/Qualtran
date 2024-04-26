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
r"""The Givens rotation Bloqs help count costs for similarity transforming
fermionic ladder operators to produce linear combinations of fermionic ladder operators.

Following notation from Reference [1] we note that a single 
ladder operator can be similarity transformed by a basis rotation to produce a linear 
combination of ladder operators
$$
U(Q)a_{q}U(Q)^{\dagger} = \sum_{p}Q_{pq}^{*}a_{p} = \overrightarrow{a}_{q}\\
U(Q)a_{q}^{\dagger}U(Q)^{\dagger} = \sum_{p}Q_{pq}a_{p}^{\dagger} = 
\overrightarrow{a}_{q}^{\dagger}
$$
Each vector of operators can be implemented by a $N$ (size of basis) Givens rotation unitaries as
$$
V_{\overrightarrow{Q}_{q}} a_{0} V_{\overrightarrow{Q}_{q}}^{\dagger} = 
\overrightarrow{a}_{q} \\
V_{\overrightarrow{Q}_{q}} a_{0}^{\dagger} V_{\overrightarrow{Q}_{q}}^{\dagger} = 
\overrightarrow{a}_{q}^{\dagger}
$$
where 
$$
V_{\overrightarrow{Q}_{q}} = V_{n-1,n-2}(0, \phi_{n-1}) V_{n-2, n-3}(\theta_{n-2}, \phi_{n-2})
V_{n-3,n-4}(\theta_{n-2}, \phi_{n-2})...V_{2, 1}(\theta_{1}, \phi_{1})
V_{1, 0}(\theta_{0}, \phi_{0})
$$
with each $V_{ij}(\theta, \phi) = \mathrm{RZ}_{j}(\pi)\mathrm{R}_{ij}(\theta)$. 
and $1$ Rz rotation for real valued $\overrightarrow{Q}$.


References:
  1.  Vera von Burg, Guang Hao Low, Thomas H ̈aner, Damian S. Steiger, Markus Reiher, 
      Martin Roetteler, and Matthias Troyer, “Quantum computing enhanced computational catalysis,” 
      Phys. Rev. Res. 3, 033055 (2021).

"""
from functools import cached_property
from typing import Dict

from attrs import frozen

from qualtran import Bloq, bloq_example, BloqBuilder, BloqDocSpec, QBit, QFxp, Signature, SoquetT
from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, XGate, Toffoli
from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

class RzAddIntoPhaseGradient(AddIntoPhaseGrad):
    r"""Temporary controlled adder to give the right complexity for Rz rotations by 
    phase gradient state addition.

    References:
         [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](
            https://arxiv.org/abs/2007.07391).
        Section II-C: Oracles for phasing by cost function. Appendix A: Addition for controlled
        rotations
    """
    def bloq_counts(self,):
        return {Toffoli(): self.x_bitsize - 2}

@frozen
class RealGivensRotationByPhaseGradient(Bloq):
    r"""Givens rotation corresponding to a 2-fermion mode transformation generated by

    $$
        e^{\theta (a_{i}^{\dagger}a_{j} - a_{j}^{\dagger}a_{i})} =  e^{i \theta (YX + XY) / 2}
    $$

    corresponding to the circuit

        i: ───X───X───S^-1───X───Rz(theta)───X───X───@───────X───S^-1───
                  │          │               │       │       │
        j: ───S───@───H──────@───Rz(theta)───@───────X───H───@──────────

    The rotation is performed by addition into a phase state and the fractional binary for
    $\theta$ is stored in an additional register.

    The Toffoli cost for this block comes from the cost of two rotations by addition into
    the phase gradient state which which is $2(b_{\mathrm{grad}}-2)$ where $b_{\mathrm{grad}}$
    is the size of the phasegradient register.

    Args:
        phasegrad_bitsize int: size of phase gradient which is also the size of the register
            representing the binary fraction of the rotation angle
    Registers:
        target_i: 1st-qubit QBit type register
        target_j: 2nd-qubit Qbit type register
        rom_data: QFxp data representing fractional binary for real part of rotation
        phase_gradient: QFxp data type representing the phase gradient register

    References:
        [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](
            https://arxiv.org/abs/2007.07391).
        Section II-C: Oracles for phasing by cost function. Appendix A: Addition for controlled
        rotations
    """
    phasegrad_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            target_i=QBit(),
            target_j=QBit(),
            rom_data=QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False),
            phase_gradient=QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False),
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        target_i: SoquetT,
        target_j: SoquetT,
        rom_data: SoquetT,
        phase_gradient: SoquetT,
    ) -> Dict[str, SoquetT]:
        # set up rz-rotation via phase-gradient state
        add_into_phasegrad_gate = RzAddIntoPhaseGradient(
            x_bitsize=self.phasegrad_bitsize,
            phase_bitsize=self.phasegrad_bitsize,
            right_shift=0,
            sign=1,
            controlled=1,
        )

        # clifford block
        target_i = bb.add(XGate(), q=target_i)
        target_j = bb.add(SGate(), q=target_j)
        target_j, target_i = bb.add(CNOT(), ctrl=target_j, target=target_i)
        target_j = bb.add(Hadamard(), q=target_j)
        target_i = bb.add(SGate(is_adjoint=True), q=target_i)
        target_j, target_i = bb.add(CNOT(), ctrl=target_j, target=target_i)

        # parallel rz (Can probably be improved with single out of place adder into a single ancilla
        target_i, rom_data, phase_gradient = bb.add(
            add_into_phasegrad_gate, x=rom_data, phase_grad=phase_gradient, ctrl=target_i
        )
        target_j, rom_data, phase_gradient = bb.add(
            add_into_phasegrad_gate, x=rom_data, phase_grad=phase_gradient, ctrl=target_j
        )

        # clifford block
        target_j, target_i = bb.add(CNOT(), ctrl=target_j, target=target_i)
        target_i = bb.add(XGate(), q=target_i)
        target_i, target_j = bb.add(CNOT(), ctrl=target_i, target=target_j)
        target_j = bb.add(Hadamard(), q=target_j)
        target_j, target_i = bb.add(CNOT(), ctrl=target_j, target=target_i)
        target_i = bb.add(SGate(), q=target_i)

        return {
            'target_i': target_i,
            'target_j': target_j,
            'rom_data': rom_data,
            'phase_gradient': phase_gradient,
        }


@bloq_example
def _real_givens() -> RealGivensRotationByPhaseGradient:
    r_givens = RealGivensRotationByPhaseGradient(phasegrad_bitsize=4)
    return r_givens


_REAL_GIVENS_DOC = BloqDocSpec(
    bloq_cls=RealGivensRotationByPhaseGradient,
    import_line='from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import RealGivensRotationByPhaseGradient',
    examples=(_real_givens,),
)


@frozen
class ComplexGivensRotationByPhaseGradient(Bloq):
    r"""Complex Givens rotation corresponding to a 2-fermion mode transformation generated by

    $$
        e^{i \phi n_{j}}e^{\theta (a_{i}^{\dagger}a_{j} - a_{j}^{\dagger}a_{i})} =  e^{i \phi Z_{j}/2}e^{i \theta (YX + XY) / 2}
    $$

    corresponding to the circuit

        i: ───X───X───S^-1───X───Rz(theta)───X───X───@───────X──S^-1─────
                  │          │               │       │       │
        j: ───S───@───H──────@───Rz(theta)───@───────X───H───@──Rz(phi)──

    The rotation is performed by addition into a phase state and the fractional binary for
    $\theta$ is stored in an additional register.

    Args:
        phasegrad_bitsize int: size of phase gradient which is also the size of the register
            representing the binary fraction of the rotation angles
    Registers:
        target_i: 1st-qubit QBit type register
        target_j: 2nd-qubit Qbit type register
        real_rom_data: QFxp data representing fractional binary for real part of rotation
        cplx_rom_data: QFxp data representing fractional binary for imag part of rotation
        phase_gradient: QFxp data type representing the phase gradient register
    """

    phasegrad_bitsize: int

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            target_i=QBit(),
            target_j=QBit(),
            real_rom_data=QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False),
            cplx_rom_data=QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False),
            phase_gradient=QFxp(self.phasegrad_bitsize, self.phasegrad_bitsize, signed=False),
        )

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        target_i: SoquetT,
        target_j: SoquetT,
        real_rom_data: SoquetT,
        cplx_rom_data: SoquetT,
        phase_gradient: SoquetT,
    ) -> Dict[str, SoquetT]:
        real_givens_gate = RealGivensRotationByPhaseGradient(
            phasegrad_bitsize=self.phasegrad_bitsize
        )

        # real-valued Givens rotation
        target_i, target_j, real_rom_data, phase_gradient = bb.add(
            real_givens_gate,
            target_i=target_i,
            target_j=target_j,
            rom_data=real_rom_data,
            phase_gradient=phase_gradient,
        )

        # set up rz-rotation on j-bit by phase-gradient state
        add_into_phasegrad_gate = RzAddIntoPhaseGradient(
            x_bitsize=self.phasegrad_bitsize,
            phase_bitsize=self.phasegrad_bitsize,
            right_shift=0,
            sign=1,
            controlled=1,
        )
        target_j, cplx_rom_data, phase_gradient = bb.add(
            add_into_phasegrad_gate, x=cplx_rom_data, phase_grad=phase_gradient, ctrl=target_j
        )
        return {
            'target_i': target_i,
            'target_j': target_j,
            'real_rom_data': real_rom_data,
            'cplx_rom_data': cplx_rom_data,
            'phase_gradient': phase_gradient,
        }


@bloq_example
def _cplx_givens() -> ComplexGivensRotationByPhaseGradient:
    c_givens = ComplexGivensRotationByPhaseGradient(phasegrad_bitsize=4)
    return c_givens


_CPLX_GIVENS_DOC = BloqDocSpec(
    bloq_cls=ComplexGivensRotationByPhaseGradient,
    import_line='from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import ComplexGivensRotationByPhaseGradient',
    examples=(_cplx_givens,),
)
