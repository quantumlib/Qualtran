"""
This file contains useful utilties for constructing the
necessary information for qubitization resource estimates
of the Hubbard model.  See PhysRevX.8.041015 for more
details

Need to implement select register Eq. 58 in PRX
"""
import cirq

import openfermion as of
from openfermion.hamiltonians.hubbard import fermi_hubbard
from collections import defaultdict


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
    norb = x_dimension * y_dimension
    if not spinless:
        norb *= 2
    qubits = cirq.LineQubit.range(norb)
    # convert to cirq.PauliString Hamiltonian
    qubit_map = dict(zip(range(norb), qubits))
    cirq_pauli_terms = []
    for term, val in jw_ham.terms.items():
        pauli_term_dict = dict([(qubit_map[xx], yy) for xx, yy in term])
        pauli_term = cirq.PauliString(pauli_term_dict, coefficient=val)
        cirq_pauli_terms.append(pauli_term.dense(qubits))
    return cirq_pauli_terms


if __name__ == "__main__":
    paulisum_hub_ham = qubit_hamiltonian_fermi_hubbard(
        x_dimension=4, y_dimension=4, tunneling=1, coulomb=4, periodic=True
    )
    d = defaultdict(lambda: [])
    for ps in paulisum_hub_ham:
        d[ps.coefficient].append(ps.copy(coefficient=1))
    for coefficient, terms in d.items():
        print(coefficient)
        print(*sorted(str(t) for t in terms))
    # print(set(ps.coefficient for ps in paulisum_hub_ham))
