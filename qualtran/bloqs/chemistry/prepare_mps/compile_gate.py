from typing import Dict, Tuple, List, Optional
from functools import cached_property
import attrs

import numpy as np

from qualtran import Bloq, BloqBuilder, Signature, SoquetT, Register, BoundedQUInt
from qualtran.bloqs.reflection_using_prepare import ReflectionUsingPrepare
from qualtran.bloqs.select_and_prepare import SelectOracle
from qualtran.bloqs.state_preparation.state_preparation_via_rotation import StatePreparationViaRotations
from qualtran.bloqs.basic_gates import Hadamard, ZGate, CNOT
from qualtran.bloqs.basic_gates.x_basis import XGate
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


@attrs.frozen
class DecomposeGateViaHR(Bloq):
    r"""
    """
    phase_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    gate_cols: Tuple[int, Tuple[complex, ...]] # tuple with the columns of the gate that are specified
    uncompute: bool = False
    internal_phase_grad: bool = False
    internal_refl_ancilla: bool = True
    control_val: Optional[int] = None

    def __attrs_post_init__(self):
        # at least one column has to be specified
        assert len(self.gate_cols) > 0
        # there can't be more columns that rows
        assert len(self.gate_cols) <= len(self.gate_cols[0][1])
        # all cols must be the same length and a power of two
        lengths = set([len(c[1]) for c in self.gate_cols])
        assert len(lengths) == 1
        assert list(lengths)[0] == 2**self.gate_bitsize

    @property
    def signature(self):
        return Signature.build(refl_ancilla=(not self.internal_refl_ancilla), gate_input=self.gate_bitsize, phase_grad=(not self.internal_phase_grad)*self.phase_bitsize)
    
    @property
    def gate_bitsize(self):
        return (len(self.gate_cols[0][1])-1).bit_length()

    def build_composite_bloq(self, bb: BloqBuilder, *, gate_input: SoquetT, **soqs: SoquetT) -> Dict[str, SoquetT]:
        """Extra soquets inside soqs are:
            * phase_grad: a phase gradient state of size phase_bitsize if internal_phase_gradient
                        is set to False
            * refl_ancilla: a clean qubit in |0> if internal_refl_ancilla is set to False
        """
        if self.internal_refl_ancilla:
            refl_ancilla = bb.allocate(1)
        else:
            refl_ancilla = soqs.pop("refl_ancilla")
        if self.internal_phase_grad:
            phase_grad = bb.add(PhaseGradientState(self.phase_bitsize))
        else:
            phase_grad = soqs.pop("phase_grad")

        if not self.uncompute:
            refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        if self.gate_bitsize == 1:
            reflection_reg = bb.join(np.array([refl_ancilla, gate_input]))
        else:
            reflection_reg = bb.join(np.array([refl_ancilla, *bb.split(gate_input)]))
        # If uncompute iterate backwards. In theory this would not make a difference, but as the
        # column compilations are approximate if done otherwise then U*U^t != I
        for i in list(range(0,len(self.gate_cols)))[::(1-2*self.uncompute)]:
            reflection_reg, phase_grad = self._ith_reflection(bb, i, reflection_reg, phase_grad)
        qubits = bb.split(reflection_reg)
        if self.uncompute:
            qubits[0] = bb.add(XGate(), q=qubits[0])
        
        soqs["gate_input"] = bb.join(qubits[1:])
        if self.internal_refl_ancilla:
            bb.free(qubits[0])
        else:
            soqs["refl_ancilla"] = qubits[0]
        if self.internal_phase_grad:
            bb.add(PhaseGradientState(self.phase_bitsize).adjoint(), phase_grad=phase_grad)
        else:
            soqs["phase_grad"] = phase_grad
        return soqs 
    
    def _ith_reflection(self, bb: BloqBuilder, i: int, reflection_reg: SoquetT, phase_grad: SoquetT):
        reflection_prep = PrepareOracleDecomposeeGateReflection(state_coefs=self.gate_cols[i][1], phase_bitsize=self.phase_bitsize, index=self.gate_cols[i][0])
        refl_bloq = ReflectionUsingPrepare(prepare_gate=reflection_prep, extra_registers=(("phase_grad", self.phase_bitsize),), control_val=self.control_val)
        reflection_reg, phase_grad = bb.add(refl_bloq, target_reg=reflection_reg, phase_grad=phase_grad)
        return reflection_reg, phase_grad


@attrs.frozen
class PrepareOracleDecomposeeGateReflection(SelectOracle):
    state_coefs: Tuple # state |u_i>
    phase_bitsize: int # number of ancilla qubits used to encode the state preparation's rotations
    index: int # i value in |i>
    uncompute: bool = False
    target_registers: Tuple[Register, ...] = ()
    junk_registers: Tuple[Register, ...] = ()
    
    @property
    def state_bitsize(self):
        return (len(self.state_coefs)-1).bit_length()
    
    @cached_property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (
            Register('target_reg', BoundedQUInt(bitsize=self.state_bitsize+1, iteration_length=2**(self.state_bitsize))),
        )
    
    @cached_property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()
    
    @property
    def signature (self):
        return Signature.build(target_reg=self.state_bitsize+1, phase_grad=self.phase_bitsize)

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
        for i, bit in enumerate(f"{self.index:0{self.state_bitsize}b}"):
            if bit == '1':
                refl_ancilla, state[i] = bb.add(CNOT(), ctrl=refl_ancilla, target=state[i])
        return refl_ancilla, state
    
    def _prepare_u_state(
            self, bb: BloqBuilder, refl_ancilla: SoquetT, state: List[SoquetT], phase_grad: SoquetT
    ):
        csp = StatePreparationViaRotations(phase_bitsize=self.phase_bitsize, state_coefficients=self.state_coefs, control_bitsize=1, uncompute=self.uncompute)
        # for negative controlling on state preparation
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        state_reg = bb.join(state)
        refl_ancilla, state_reg, phase_grad = bb.add(csp, prepare_control=refl_ancilla, target_state=state_reg, phase_gradient=phase_grad)
        state = bb.split(state_reg)
        refl_ancilla = bb.add(XGate(), q=refl_ancilla)
        return refl_ancilla, state, phase_grad