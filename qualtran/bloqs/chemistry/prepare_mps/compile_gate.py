from typing import Dict, Tuple
import attrs


from qualtran import Bloq, BloqBuilder, SelectionRegister, Signature, SoquetT
from qualtran.bloqs.controlled_state_preparation import ControlledStatePreparationUsingRotations
from qualtran.bloqs.select_and_prepare import PrepareOracle


@attrs.frozen
class CompileGateGivenVectors(Bloq):
    n_qubits: int  # number of qubits that the gate acts on
    gate_coefs: Tuple[Tuple] # tuple with the columns/rows of the gate that are specified
    adjoint: bool = False

    @property
    def signature(self):
        return Signature.build(x=self.n_qubits)

    def build_composite_bloq(
        self, bb: BloqBuilder, *, x: SoquetT
    ) -> Dict[str, SoquetT]:
            
        return {"x": x}


@attrs.frozen
class PrepareOracleCompileGateReflection(PrepareOracle):
    n_qubits: int # length in qubits of the state |u_i> (without the reflection ancilla!)
    state: Tuple # state |u_i>
    index: int # i value in |i>

    @property
    def selection_registers(self) -> Tuple[SelectionRegister, ...]:
        return (
            SelectionRegister(
                "state", bitsize=self.n_qubits+1, iteration_length=self.n_qubits+1
            ),
        )
    
    