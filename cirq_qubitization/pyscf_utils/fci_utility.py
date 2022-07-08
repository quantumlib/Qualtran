"""
Helper functions for getting FCI coefficients
"""
from typing import List
import numpy as np
from pyscf import gto, scf, fci
from pyscf.fci.cistring import make_strings
import fqe


def pyscf_to_fqe_wf(pyscf_cimat: np.ndarray, pyscf_mf=None, norbs=None, nelec=None) -> fqe.Wavefunction:
    if pyscf_mf is None:
        assert norbs is not None
        assert nelec is not None
    else:
        mol = pyscf_mf.mol
        nelec = mol.nelec
        norbs = pyscf_mf.mo_coeff.shape[1]

    # Get alpha and beta strings
    norb_list = tuple(list(range(norbs)))
    n_alpha_strings = [x for x in make_strings(norb_list, nelec[0])]
    n_beta_strings = [x for x in make_strings(norb_list, nelec[1])]

    # get fqe Wavefunction object to populate
    fqe_wf_ci = fqe.Wavefunction([[sum(nelec), nelec[0] - nelec[1], norbs]])
    fqe_data_ci = fqe_wf_ci.sector((sum(nelec), nelec[0] - nelec[1]))
    fqe_graph_ci = fqe_data_ci.get_fcigraph()

    # get coeff mat to populate Wavefunction object
    fqe_orderd_coeff = np.zeros((fqe_graph_ci.lena(), fqe_graph_ci.lenb()),
                                dtype=np.complex128)  # only works for complex128 right now
    for paidx, pyscf_alpha_idx in enumerate(n_alpha_strings):
        for pbidx, pyscf_beta_idx in enumerate(n_beta_strings):
            fqe_orderd_coeff[fqe_graph_ci.index_alpha(pyscf_alpha_idx), 
                             fqe_graph_ci.index_beta(pyscf_beta_idx)] = \
                                pyscf_cimat[paidx, pbidx]

    # populate Wavefunction object
    fqe_data_ci.coeff = fqe_orderd_coeff
    return fqe_wf_ci


def get_fqe_wfns(pyscf_ci_wfns: List[np.ndarray], pyscf_mf: scf.RHF) -> List[fqe.Wavefunction]:
    """
    Construct FQE wavefunctions from pyscf wavefunctions

    :param pyscf_ci_wfns: List of numpy arrays that represent the pyscf wavefunctions
    :param pyscf_mf: pyscf RHF mean-field object
    """
    fqe_wfns = []
    for ci_wfn in pyscf_ci_wfns:
        fqe_wfn = pyscf_to_fqe_wf(ci_wfn, pyscf_mf)
        fqe_wfns.append(fqe_wfn)
    return fqe_wfns


def get_spectrum(mf: scf.RHF, num_roots: int):
    """
    Get spectrum of molecule defined as RHF object

    :param mf: psycf mean-field object
    :param num_roots: number of roots to get from fci
    """
    myci = fci.FCI(mf)
    roots, wfs = myci.kernel(nroots=num_roots)
    return roots, wfs

