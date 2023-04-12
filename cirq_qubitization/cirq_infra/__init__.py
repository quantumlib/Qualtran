from cirq_qubitization.cirq_infra.decompose_protocol import decompose_once_into_operations
from cirq_qubitization.cirq_infra.gate_with_registers import GateWithRegisters, Register, Registers
from cirq_qubitization.cirq_infra.qid_types import BorrowableQubit, CleanQubit
from cirq_qubitization.cirq_infra.qubit_management_transformers import (
    map_clean_and_borrowable_qubits,
)
from cirq_qubitization.cirq_infra.qubit_manager import (
    GreedyQubitManager,
    memory_management_context,
    qalloc,
    qborrow,
    qfree,
    SimpleQubitManager,
)
