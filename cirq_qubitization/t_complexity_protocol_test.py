import cirq
import pytest
from cirq_qubitization.t_complexity_protocol import TComplexity, t_complexity
from cirq_qubitization.and_gate import And


class SupportTComplexity(cirq.Operation):
    
    def qubits(self):
        return []
    
    def with_qubits(self, _):
        pass

    def _t_complexity_(self) -> TComplexity:
        return TComplexity(t=1)        

class DoesNotSupportTComplexityButSupportsDecomposition:
        
    def _decompose_(self):
        yield SupportTComplexity(), SupportTComplexity()

class DoesNotSupportTComplexity:
    pass


def test_t_complexity():
    with pytest.raises(TypeError):
        _ = t_complexity(DoesNotSupportTComplexity())

    assert t_complexity(SupportTComplexity()) == TComplexity(t=1)

    assert t_complexity(DoesNotSupportTComplexityButSupportsDecomposition()) == TComplexity(t=2)

    gate = And()
    op = gate.on_registers(**gate.registers.get_named_qubits())
    assert t_complexity(op) == TComplexity(t=4, clifford=9)

    gate = And() ** -1
    op = gate.on_registers(**gate.registers.get_named_qubits())
    assert t_complexity(op) == TComplexity(clifford=4)