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
    n_qubits: int  # number of qubits that the gate acts on
    rot_reg_size: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_coefs: Tuple[Tuple] # tuple with the columns/rows of the gate that are specified
    adjoint: bool = False

    @property
    def signature(self):
        return Signature.build(gate_input=self.n_qubits)
    
    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT) -> Dict[str, SoquetT]:
        gate_compiler = CompileGateGivenVectors(n_qubits=self.n_qubits, gate_coefs=self.gate_coefs, rot_reg_size=self.rot_reg_size, adjoint=self.adjoint)
        phase_gradient = bb.add(PhaseGradientState(bitsize=self.rot_reg_size))
        gate_input, phase_gradient = bb.add(gate_compiler, gate_input=gate_input, phase_grad=phase_gradient)
        bb.add(PhaseGradientState(bitsize=self.rot_reg_size, adjoint=True), phase_grad=phase_gradient)
        return {"gate_input": gate_input}


@attrs.frozen
class CompileGateGivenVectors(Bloq):
    n_qubits: int  # number of qubits that the gate acts on
    rot_reg_size: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_coefs: Tuple[Tuple] # tuple with the columns of the gate that are specified
    adjoint: bool = False

    @property
    def signature(self):
        return Signature.build(gate_input=self.n_qubits, phase_grad=self.rot_reg_size)

    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT, phase_grad: SoquetT) -> Dict[str, SoquetT]:
        if self.adjoint:
            reflection_reg = bb.join(np.array([bb.add(ZeroState()), *bb.split(gate_input)]))
        else:
            reflection_reg = bb.join(np.array([bb.add(OneState()), *bb.split(gate_input)]))
        for i in range(len(self.gate_coefs)):
            reflection_reg, phase_grad = self.__ithReflection(bb, i, reflection_reg, phase_grad)
        qubits = bb.split(reflection_reg)
        if self.adjoint:
            bb.add(OneEffect(), q=qubits[0])
        else:
            bb.add(ZeroEffect(), q=qubits[0])
        gate_input = bb.join(qubits[1:])
        return {"gate_input": gate_input, "phase_grad": phase_grad}
    
    def __ithReflection(self, bb: BloqBuilder, i: int, reflection_reg: SoquetT, phase_grad: SoquetT):
        reflection_prep = PrepareOracleCompileGateReflection(n_qubits=self.n_qubits, state_coefs=self.gate_coefs[i], rot_reg_size=self.rot_reg_size, index=i)
        reflection_prep_adj = PrepareOracleCompileGateReflection(n_qubits=self.n_qubits, state_coefs=self.gate_coefs[i], rot_reg_size=self.rot_reg_size, index=i, adjoint=True)
        reflection_reg, phase_grad = bb.add(reflection_prep_adj, target_reg=reflection_reg, phase_grad=phase_grad)
        reflection_reg = self.__reflect(bb, reflection_reg)
        reflection_reg, phase_grad = bb.add(reflection_prep, target_reg=reflection_reg, phase_grad=phase_grad)
        return reflection_reg, phase_grad
    
    def __reflect(self, bb: BloqBuilder, reg: SoquetT):
        mult_control_flip = MultiControlPauli(cvs=tuple([0]*(self.n_qubits+1)), target_gate=cirq.Z)
        ancilla = bb.add(OneState())
        reg, ancilla = bb.add(mult_control_flip, controls=reg, target=ancilla)
        bb.add(OneEffect(), q=ancilla)
        return reg


@attrs.frozen
class PrepareOracleCompileGateReflection(PrepareOracle):
    n_qubits: int # length in qubits of the state |u_i> (without the reflection ancilla!)
    state_coefs: Tuple # state |u_i>
    rot_reg_size: int # number of ancilla qubits used to encode the state preparation's rotations
    index: int # i value in |i>
    adjoint: bool = False

    @property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "target_reg", bitsize=self.n_qubits+1, iteration_length=self.n_qubits+1
            ),
        )
    
    @property
    def signature (self):
        return Signature.build(target_reg=self.n_qubits+1, phase_grad=self.rot_reg_size)

    def build_composite_bloq(
            self, bb: BloqBuilder, *, target_reg: SoquetT, phase_grad: SoquetT
    ) -> Dict[str, SoquetT]:
        qubits = bb.split(target_reg)
        refl_ancilla = qubits[0]
        state = qubits[1:]
        # if the gate is the adjoint, the |u> and |i> states should be prepared first and then the
        # ancilla, if it is the normal version the ancilla is prepared first and then |u> and |i>
        if not self.adjoint:
            refl_ancilla = self.__prepareReflectionAncilla(bb, refl_ancilla)
        refl_ancilla, state = self.__prepareIState(bb, refl_ancilla, state)
        refl_ancilla, state, phase_grad = self.__prepareUState(bb, refl_ancilla, state, phase_grad)
        if self.adjoint:
            refl_ancilla = self.__prepareReflectionAncilla(bb, refl_ancilla)
        qubits[0] = refl_ancilla
        qubits[1:] = state
        target_reg = bb.join(qubits)
        return {"target_reg": target_reg, "phase_grad": phase_grad}
    
    def __prepareReflectionAncilla (self, bb: BloqBuilder, refl_ancilla: SoquetT):
        # prepare/unprepare the ancilla from |0> to 1/sqrt(2)(|1> - |0>)
        refl_ancilla = bb.add(Hadamard(), q=refl_ancilla)
        refl_ancilla = bb.add(ZGate(), q=refl_ancilla)
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        return refl_ancilla
    
    def __prepareIState(self, bb: BloqBuilder, refl_ancilla: SoquetT, state: List[SoquetT]):
        for i, bit in enumerate(f"{self.index:0{self.n_qubits}b}"):
            if bit == '1':
                refl_ancilla, state[i] = bb.add(CNOT(), ctrl=refl_ancilla, target=state[i])
        return refl_ancilla, state
    
    def __prepareUState(
            self, bb: BloqBuilder, refl_ancilla: SoquetT, state: List[SoquetT], phase_grad: SoquetT
    ):
        csp = ControlledStatePreparationUsingRotations(n_qubits=self.n_qubits, rot_reg_size=self.rot_reg_size, state=self.state_coefs, adjoint=self.adjoint)
        # for negative controlling on state preparation
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        state_reg = bb.join(state)
        refl_ancilla, state_reg, phase_grad = bb.add(csp, control=refl_ancilla, target_state=state_reg, phase_gradient=phase_grad)
        state = bb.split(state_reg)
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        return refl_ancilla, state, phase_grad