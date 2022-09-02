from typing import Sequence

import cirq
import openfermion as of
from openfermion.hamiltonians.hubbard import fermi_hubbard

from cirq_qubitization.apply_gate_to_lth_target import ApplyGateToLthQubit
from cirq_qubitization.gate_with_registers import Registers, GateWithRegisters
from cirq_qubitization.selected_majorana_fermion import SelectedMajoranaFermionGate
from cirq_qubitization.swap_network import MultiTargetCSwap


def qubit_hamiltonian_fermi_hubbard(
    x_dimension,
    y_dimension,
    tunneling,
    coulomb,
    chemical_potential=0.0,
    magnetic_field=0.0,
    periodic=True,
    spinless=False,
    particle_hole_symmetry=False,
):
    # use OpenFermion utility to obtain Fermi-Hubbard Model
    hub_ham = fermi_hubbard(
        x_dimension,
        y_dimension,
        tunneling,
        coulomb,
        chemical_potential=chemical_potential,
        magnetic_field=magnetic_field,
        periodic=periodic,
        spinless=spinless,
        particle_hole_symmetry=particle_hole_symmetry,
    )
    jw_ham = of.jordan_wigner(hub_ham)
    return of.transforms.qubit_operator_to_pauli_sum(jw_ham)


class SelectHubbard(GateWithRegisters):
    def __init__(self, x_dim: int, y_dim: int):
        self.x_dim = x_dim
        self.y_dim = y_dim

    @property
    def registers(self) -> Registers:
        x_size = (self.x_dim - 1).bit_length()
        y_size = (self.y_dim - 1).bit_length()
        N = self.x_dim * self.y_dim * 2
        return Registers.build(
            control=1,
            U=1,
            V=1,
            p_x=x_size,
            p_y=y_size,
            alpha=1,
            q_x=x_size,
            q_y=y_size,
            beta=1,
            target=N,
            ancilla=x_size + y_size + 2,
        )

    def decompose_from_registers(
        self,
        control: Sequence[cirq.Qid],
        U: Sequence[cirq.Qid],
        V: Sequence[cirq.Qid],
        p_x: Sequence[cirq.Qid],
        p_y: Sequence[cirq.Qid],
        alpha: Sequence[cirq.Qid],
        q_x: Sequence[cirq.Qid],
        q_y: Sequence[cirq.Qid],
        beta: Sequence[cirq.Qid],
        target: Sequence[cirq.Qid],
        ancilla: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        (control,) = control
        (U,) = U
        (V,) = V
        (alpha,) = alpha
        (beta,) = beta
        yield SelectedMajoranaFermionGate(
            len(p_x) + len(p_y) + 1, len(target), target_gate=cirq.Y
        ).on_registers(
            control=control,
            selection=p_x + p_y + (alpha,),
            ancilla=ancilla[:-1],
            accumulator=ancilla[-1],
            target=target,
        )
        yield MultiTargetCSwap(len(p_x) + len(p_y) + 1).on_registers(
            control=V, target_x=p_x + p_y + (alpha,), target_y=q_x + q_y + (beta,)
        )
        yield SelectedMajoranaFermionGate(
            len(q_x) + len(q_y) + 1, len(target), target_gate=cirq.X
        ).on_registers(
            control=control,
            selection=q_x + q_y + (beta,),
            ancilla=ancilla[:-1],
            accumulator=ancilla[-1],
            target=target,
        )
        yield MultiTargetCSwap(len(p_x) + len(p_y) + 1).on_registers(
            control=V, target_x=p_x + p_y + (alpha,), target_y=q_x + q_y + (beta,)
        )
        yield cirq.S(control) ** -1
        yield cirq.CZ(control, U)
        yield ApplyGateToLthQubit(
            selection_bitsize=len(q_x) + len(q_y) + 1,
            target_bitsize=len(target),
            nth_gate=lambda n: cirq.Z if n & 1 else cirq.I,
            control_bitsize=2,
        ).on_registers(
            control=[control, V],
            selection=q_x + q_y + (beta,),
            ancilla=ancilla[:],  # FIXME: accumulator is last one
            target=target,
        )
