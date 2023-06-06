import itertools

import cirq
import numpy as np

import cirq_qubitization as cq
import cirq_qubitization.cirq_infra.testing as cq_testing
from cirq_qubitization.bit_tools import iter_bits
from cirq_qubitization.cirq_algos.chemistry import (
    IndexedAddMod,
    PrepareChem,
    SelectChem,
    SubPrepareChem,
)
from cirq_qubitization.cirq_infra.gate_with_registers import SelectionRegisters
from cirq_qubitization.jupyter_tools import execute_notebook


def test_select_t_complexity():
    M = 2
    select = SelectChem(M=M, control_val=1)
    cost = cq.t_complexity(select)


# def test_indexed_add_mod():
#     greedy_mm = cq.cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
#     target_shape = [2, 2, 2]
#     mod = 2
#     with cq.cirq_infra.memory_management_context(greedy_mm):
#         gate = IndexedAddMod(target_shape, mod)
#         g = cq_testing.GateHelper(gate)
#         assert len(g.all_qubits) <= gate.registers.bitsize + gate.selection_registers.bitsize - 1


#     max_i, max_j, max_k = target_shape
#     i_len, j_len, k_len = tuple(reg.bitsize for reg in gate.selection_registers)
#     print()
#     for i, j, k in itertools.product(range(max_i), range(max_j), range(max_k)):
#         qubit_vals = {x: 0 for x in g.all_qubits}
#         print("QB: ", qubit_vals, g.quregs['t1'])
#         # Initialize selection bits appropriately:
#         qubit_vals.update(zip(g.quregs['i'], iter_bits(i, i_len)))
#         qubit_vals.update(zip(g.quregs['j'], iter_bits(j, j_len)))
#         qubit_vals.update(zip(g.quregs['k'], iter_bits(k, k_len)))
#         qubit_vals.update(zip(g.quregs['t1'], iter_bits(i, i_len)))
#         qubit_vals.update(zip(g.quregs['t2'], iter_bits(j, j_len)))
#         qubit_vals.update(zip(g.quregs['t3'], iter_bits(k, k_len)))
#         # Construct initial state
#         initial_state = [qubit_vals[x] for x in g.all_qubits]
#         print("IJK: ", i, j, k, initial_state)
#         for idx, (reg_name, idx_val) in enumerate(zip(['t1', 't2', 't3'], [i, j, k])):
#             print(idx, reg_name, idx_val, (2 * idx_val) % gate.mod)
#             qubit_vals.update(
#                 zip(g.quregs[reg_name], iter_bits((2 * idx_val) % gate.mod, target_shape[idx]))
#             )
#         final_state = [qubit_vals[x] for x in g.all_qubits]
#         act, sb = cq_testing.get_circuit_inp_out_cirqsim(
#             g.decomposed_circuit, g.all_qubits, initial_state, final_state
#         )
#         cq_testing.assert_circuit_inp_out_cirqsim(
#             g.decomposed_circuit, g.all_qubits, initial_state, final_state
#         )


# def test_indexed_add_mod():
#     greedy_mm = cq.cirq_infra.GreedyQubitManager(prefix="_a", maximize_reuse=True)
#     selection_bitsize = 3
#     mod = 3
#     with cq.cirq_infra.memory_management_context(greedy_mm):
#         gate = IndexedAddMod(selection_bitsize, mod)

#     g = cq_testing.GateHelper(gate)
#     maps = {}
#     add_val = 2  # stored in target register
#     bitsize = 2
#     for sel in range(2**bitsize):
#         inp = f'0b_{sel:0{bitsize}b}_{add_val:0{bitsize}b}'
#         y = (sel + add_val) % mod if sel < mod else sel
#         out = f'0b_{sel:0{bitsize}b}_{y:0{bitsize}b}'
#         maps[int(inp, 2)] = int(out, 2)
#     num_qubits = gate.num_qubits()
#     op = gate.on(*cirq.LineQubit.range(num_qubits))
#     circuit = cirq.Circuit(op)
#     cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)


def test_prepare():
    M = 2
    num_spat_orb = M**3
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * num_spat_orb).reshape((4, num_spat_orb))
    prep = PrepareChem(M=M, T=Ts, U=Us, V=Vs, Vx=Vxs)
    g = cq_testing.GateHelper(prep)
    cirq.Circuit(cirq.decompose_once(g.operation))
    cost = cq.t_complexity(prep)
