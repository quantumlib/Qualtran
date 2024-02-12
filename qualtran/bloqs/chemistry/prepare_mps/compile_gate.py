from typing import Dict, Tuple, List
import attrs

import numpy as np
import cirq

from qualtran import Bloq, BloqBuilder, SelectionRegister, Signature, SoquetT
from qualtran.bloqs.controlled_state_preparation import ControlledStatePreparationUsingRotations
from qualtran.bloqs.basic_gates import Hadamard, ZGate, CNOT
from qualtran.bloqs.multi_control_multi_target_pauli import MultiControlPauli
from qualtran.bloqs.basic_gates import ZeroState, OneState, ZeroEffect, OneEffect
from qualtran.bloqs.basic_gates.x_basis import XGate
from qualtran.bloqs.select_and_prepare import PrepareOracle
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


@attrs.frozen
class CompileGateGivenVectorsWithoutPG(Bloq):
    gate_bitsizes: int  # number of qubits that the gate acts on
    rot_reg_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_coefs: Tuple[Tuple] # tuple with the columns/rows of the gate that are specified
    uncompute: bool = False

    @property
    def signature(self):
        return Signature.build(gate_input=self.gate_bitsizes)
    
    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT) -> Dict[str, SoquetT]:
        gate_compiler = CompileGateGivenVectors(gate_bitsizes=self.gate_bitsizes, gate_coefs=self.gate_coefs, rot_reg_bitsize=self.rot_reg_bitsize, uncompute=self.uncompute)
        phase_gradient = bb.add(PhaseGradientState(bitsize=self.rot_reg_bitsize))
        gate_input, phase_gradient = bb.add(gate_compiler, gate_input=gate_input, phase_grad=phase_gradient)
        bb.add(PhaseGradientState(bitsize=self.rot_reg_bitsize, uncompute=True), phase_grad=phase_gradient)
        return {"gate_input": gate_input}


@attrs.frozen
class CompileGateGivenVectors(Bloq):
    gate_bitsizes: int  # number of qubits that the gate acts on
    rot_reg_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_coefs: Tuple[Tuple] # tuple with the columns of the gate that are specified
    uncompute: bool = False

    @property
    def signature(self):
        return Signature.build(gate_input=self.gate_bitsizes, phase_grad=self.rot_reg_bitsize)

    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT, phase_grad: SoquetT) -> Dict[str, SoquetT]:
        if self.uncompute:
            reflection_reg = bb.join(np.array([bb.add(ZeroState()), *bb.split(gate_input)]))
        else:
            reflection_reg = bb.join(np.array([bb.add(OneState()), *bb.split(gate_input)]))
        for i in range(len(self.gate_coefs)):
            reflection_reg, phase_grad = self._ith_reflection(bb, i, reflection_reg, phase_grad)
        qubits = bb.split(reflection_reg)
        if self.uncompute:
            bb.add(OneEffect(), q=qubits[0])
        else:
            bb.add(ZeroEffect(), q=qubits[0])
        gate_input = bb.join(qubits[1:])
        return {"gate_input": gate_input, "phase_grad": phase_grad}
    
    def _ith_reflection(self, bb: BloqBuilder, i: int, reflection_reg: SoquetT, phase_grad: SoquetT):
        reflection_prep = PrepareOracleCompileGateReflection(state_bitsizes=self.gate_bitsizes, state_coefs=self.gate_coefs[i], rot_reg_bitsize=self.rot_reg_bitsize, index=i)
        reflection_prep_adj = PrepareOracleCompileGateReflection(state_bitsizes=self.gate_bitsizes, state_coefs=self.gate_coefs[i], rot_reg_bitsize=self.rot_reg_bitsize, index=i, uncompute=True)
        reflection_reg, phase_grad = bb.add(reflection_prep_adj, target_reg=reflection_reg, phase_grad=phase_grad)
        reflection_reg = self.__reflect(bb, reflection_reg)
        reflection_reg, phase_grad = bb.add(reflection_prep, target_reg=reflection_reg, phase_grad=phase_grad)
        return reflection_reg, phase_grad
    
    def _reflect(self, bb: BloqBuilder, reg: SoquetT):
        mult_control_flip = MultiControlPauli(cvs=tuple([0]*(self.gate_bitsizes+1)), target_gate=cirq.Z)
        ancilla = bb.add(OneState())
        reg, ancilla = bb.add(mult_control_flip, controls=reg, target=ancilla)
        bb.add(OneEffect(), q=ancilla)
        return reg


@attrs.frozen
class PrepareOracleCompileGateReflection(PrepareOracle):
    state_bitsizes: int # length in qubits of the state |u_i> (without the reflection ancilla!)
    state_coefs: Tuple # state |u_i>
    rot_reg_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    index: int # i value in |i>
    uncompute: bool = False

    @property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "target_reg", bitsize=self.state_bitsizes+1, iteration_length=self.state_bitsizes+1
            ),
        )
    
    @property
    def signature (self):
        return Signature.build(target_reg=self.state_bitsizes+1, phase_grad=self.rot_reg_bitsize)

    def build_composite_bloq(
            self, bb: BloqBuilder, *, target_reg: SoquetT, phase_grad: SoquetT
    ) -> Dict[str, SoquetT]:
        qubits = bb.split(target_reg)
        refl_ancilla = qubits[0]
        state = qubits[1:]
        # if the gate is the adjoint, the |u> and |i> states should be prepared first and then the
        # ancilla, if it is the normal version the ancilla is prepared first and then |u> and |i>
        if not self.uncompute:
            refl_ancilla = self._prepare_reflection_ancilla(bb, refl_ancilla)
        refl_ancilla, state = self._prepare_i_state(bb, refl_ancilla, state)
        refl_ancilla, state, phase_grad = self._prepare_u_state(bb, refl_ancilla, state, phase_grad)
        if self.uncompute:
            refl_ancilla = self._prepare_reflection_ancilla(bb, refl_ancilla)
        qubits[0] = refl_ancilla
        qubits[1:] = state
        target_reg = bb.join(qubits)
        return {"target_reg": target_reg, "phase_grad": phase_grad}
    
    def _prepare_reflection_ancilla (self, bb: BloqBuilder, refl_ancilla: SoquetT):
        # prepare/unprepare the ancilla from |0> to 1/sqrt(2)(|1> - |0>)
        refl_ancilla = bb.add(Hadamard(), q=refl_ancilla)
        refl_ancilla = bb.add(ZGate(), q=refl_ancilla)
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        return refl_ancilla
    
    def _prepare_i_state(self, bb: BloqBuilder, refl_ancilla: SoquetT, state: List[SoquetT]):
        for i, bit in enumerate(f"{self.index:0{self.state_bitsizes}b}"):
            if bit == '1':
                refl_ancilla, state[i] = bb.add(CNOT(), ctrl=refl_ancilla, target=state[i])
        return refl_ancilla, state
    
    def _prepare_u_state(
            self, bb: BloqBuilder, refl_ancilla: SoquetT, state: List[SoquetT], phase_grad: SoquetT
    ):
        csp = ControlledStatePreparationUsingRotations(state_bitsizes=self.state_bitsizes, rot_reg_bitsize=self.rot_reg_bitsize, state=self.state_coefs, uncompute=self.uncompute)
        # for negative controlling on state preparation
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        state_reg = bb.join(state)
        refl_ancilla, state_reg, phase_grad = bb.add(csp, control=refl_ancilla, target_state=state_reg, phase_gradient=phase_grad)
        state = bb.split(state_reg)
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        return refl_ancilla, state, phase_grad