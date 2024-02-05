from typing import Dict, Tuple
import attrs

from qualtran import Bloq, BloqBuilder, Signature, SoquetT
from qualtran.bloqs.controlled_state_preparation import ControlledStatePreparationUsingRotations


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