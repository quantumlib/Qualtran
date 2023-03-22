import fqe
import numpy as np
from pyscf import gto, scf

from cirq_qubitization.pyscf_utils.fci_utility import get_fqe_wfns, get_spectrum


def test_fci_util():
    mol = gto.M()
    mol.atom = 'Li 0 0 0; H 0 0 1.6'
    mol.basis = '6-31g'
    mol.build()

    mf = scf.RHF(mol)
    mf.run()

    pyscf_roots, pyscf_wfs = get_spectrum(mf, num_roots=10)
    print(pyscf_roots)
    fqe_wfns = get_fqe_wfns(pyscf_wfs, mf)
    for fw in fqe_wfns:
        print()
        fw.print_wfn()

    # if we have eigenstates then they should be orthogonal
    overlap_mat = np.zeros((len(pyscf_roots), len(pyscf_roots)), dtype=np.complex128)
    for i in range(len(pyscf_roots)):
        for j in range(len(pyscf_roots)):
            overlap_mat[i, j] = fqe.vdot(fqe_wfns[i], fqe_wfns[j])
    # check if we have identity matrix
    assert np.allclose(overlap_mat, np.eye(len(pyscf_roots), dtype=np.complex128))
