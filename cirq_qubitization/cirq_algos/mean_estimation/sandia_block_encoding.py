from functools import cached_property
from math import factorial
from typing import Sequence, Tuple

import cirq
import numpy as np
from attrs import frozen
from sympy import factorint

import cirq_qubitization as cq
from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos.mean_estimation.pvec_epsmat import epsmat as eps_mt
from cirq_qubitization.cirq_algos.mean_estimation.pvec_epsmat import pvec as pv
from cirq_qubitization.cirq_algos.select_and_prepare import PrepareOracle


def M1(k, K):
    return np.ceil(np.log2(factorial(K) * np.sum([1 / factorial(k1) for k1 in range(k, K + 1)])))


def g1(x, n):
    """g1[x_, n_] := Floor[2^n*ArcSin[(1/2)/Sqrt[x]]/(2 Pi)] * 2 Pi / 2^n"""
    asin_val = np.arcsin(0.5 / np.sqrt(x))
    floored_val = np.floor(2**n * asin_val / (2 * np.pi))
    return floored_val * 2 * np.pi / 2**n


def h1(x, n):
    return x * (
        (1 + (2 - 4 * x) * np.sin(g1(x, n)) ** 2) ** 2
        + 4 * np.sin(g1(x, n)) ** 2 * np.cos(g1(x, n)) ** 2
    )


def g2(x, n):
    asin_val = np.arcsin(0.5 / np.sqrt(x))
    return np.ceil(2**n * asin_val / (2 * np.pi)) * 2 * np.pi / 2**n


def h2(x, n):
    return x * (
        (1 + (2 - 4 * x) * np.sin(g2(x, n)) ** 2) ** 2
        + 4 * np.sin(g2(x, n)) ** 2 * np.cos(g2(x, n)) ** 2
    )


def h(x, n):
    return np.max([h1(x, n), h2(x, n)])


def Eq(n, br):
    return h(n / 2 ** (np.ceil(np.log2(n))), br)


def Er(zeta):
    kt1 = 2 ** np.floor(np.log2(zeta) / 2)
    kt2 = 2 ** np.ceil(np.log2(zeta) / 2)
    return np.min([np.ceil(zeta / kt1) + kt1, np.ceil(zeta / kt2) + kt2])


@frozen
class StoppingPowerSystem:
    """Data representing the system we are stopping and projectile

    Args
        np:    number of bits in each direction for the momenta
        nn:    number of bits in each direction for the nucleus
        eta:   number of electrons
        Omega: cell volume in Bohr^3
        eps:   total allowable error
        nMc:   adjustment for the number of bits for M (used in nu preparation)
        nbr:   adjustment in the number of bits used for the nuclear positions
        L:     number of nuclei
        zeta:  charge of the projectile
        mpr:   mass of the extra nucleus
        kmean: mean momentum for the extra nucleas
    """

    n_p: int
    n_n: int
    eta: int
    Omega: float
    eps: float
    nMc: int
    nbr: int
    L: int
    zeta: int
    mpr: float
    kmean: float

    def get_system_size(self):
        return 3 * (self.eta * self.n_p + self.n_n)

    def get_kmean_in_au(self):
        projectile_ke = 0.5 * self.mpr * self.kmean**2
        return np.sqrt(2 * projectile_ke / self.mpr) * self.mpr  # p = m * v


def _pw_qubitization_with_projectile_costs_from_v5(
    stop_sys: StoppingPowerSystem,
) -> Tuple[float, float, int]:
    """Internal function for costing out time evolution operator

    Args:
        stop_sys: Stopping power system

    Returns:
        Cost of block encoding the PREPARE oracle as a tuple of
        (num_toffoli, lambda, num_logical_qubits).
    """
    n_p = stop_sys.n_p
    n_n = stop_sys.n_n
    eta = stop_sys.eta
    Omega = stop_sys.Omega
    eps = stop_sys.eps
    nMc = stop_sys.nMc
    nbr = stop_sys.nbr
    L = stop_sys.L
    zeta = stop_sys.zeta
    mpr = stop_sys.mpr
    kmean = stop_sys.get_kmean_in_au()

    # Total nuclear charge assumed to be equal to number of electrons.
    lam_zeta = eta

    # (*This is the number of bits used in rotations in preparations of equal superposition states.
    br = 7

    # Probability of success for creating the superposition over 3 basis states
    Peq0 = Eq(3, 8)

    # Probability of success for creating the superposition over i and j.
    # The extra + \[Zeta] is to account for the preparation with the extra
    # nucleus treated quantum mechanically.
    Peq1 = Eq(eta, br) ** 2

    # (*Probability of success for creating the equal superposition
    # for the selection between U and V.*)
    Peq3 = 1

    # This uses pvec from planedata.nb, which is precomputed values for
    #  \[Lambda]_\[Nu]. We start with a very large  guess for the number
    # of bits to use for M (precision in \[Nu] \ preparation) then adjust it.*)
    p = pv[n_p - 1, 49]
    pn = pv[n_n - 1, 49]

    # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64 * (2**n_p - 1)) * p / (2 * np.pi * Omega ** (1 / 3))
    tmpn = (64 * (2**n_n - 1)) * pn / (2 * np.pi * Omega ** (1 / 3))  # same but for nucleus

    # (*See Eq. (D31) or (25).*)
    # tmp*(2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta] (\[Eta] - 1 + 2 \[Zeta]))
    # For the case where there is the extra nucleus, the \[Lambda]_U has
    # \[Eta] replced with \[Eta] + \[Zeta]. For \[Lambda]_V the \[Eta] (\[Eta] - 1)
    # is replaced with (\[Eta] + \[Zeta])^2 - \[Eta] - \[Zeta]^2 = \[Eta] (\[Eta] - 1 + 2 \[Zeta]).
    # The total gives 2 (\[Eta] + \[Zeta]) \[Lambda]\[Zeta] + \[Eta]
    # (-1 + 2 \[Zeta] + \[Eta]) used here, and the \[Eta] does not factor
    # out so is not given in tmp as before
    lam_UV = tmp * (2 * eta * lam_zeta + eta * (eta - 1 + 2 * zeta)) + tmpn * 2 * zeta * lam_zeta

    # (*See Eq. (25), possibly should be replaced with expression from Eq. (71).*)
    #  Here the \[Eta] is replaced with \[Eta] + \[Zeta], because we are accounting
    # for the extra nucleus quantum mechanically. The + \[Zeta] rather than +  1 is
    # because we are using the preparation over i, j in common with the block
    # encoding of the potential, and there the charge of the nucleus is needed.
    # lam_T = 6 * (eta + zeta) * np.pi**2 / Omega**(2/3) * (2**(np - 1))**2
    lam_T = 6 * (eta + 1.0 / mpr) * np.pi**2 / Omega ** (2 / 3) * (
        2 ** (n_p - 1)
    ) ** 2 + 2 * np.pi * kmean / (mpr * Omega ** (1 / 3)) * (2 ** (n_n - 1)) ** 2 / (
        2 ** (n_n - 1) - 1
    )

    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*)
    nM = nMc + int(np.rint(np.log2(20 * lam_UV / eps)))

    #  (*Recompute p and \[Lambda]V.*)
    p = pv[n_p - 1, nM - 1]
    pn = pv[n_n - 1, nM - 1]
    lam_V = tmp * eta * (eta - 1)
    lam_Vn = tmp * eta * 2 * zeta
    lam_U = tmp * 2 * eta * lam_zeta
    lam_Un = tmpn * 2 * zeta * lam_zeta

    # (*See Eq. (117).*)
    # We will need to account for different success amplitudes for p and \
    # pn.*)
    # (*We estimate the error due to the finite M using the \
    # precomputed table.
    #   For the extra nucleus we again replace \[Eta](\[Eta]-1) with \
    # \[Eta](\[Eta]-1+2\[Zeta]).
    pamp = np.sin(3 * np.arcsin(np.sqrt(p))) ** 2
    pnmp = np.sin(3 * np.arcsin(np.sqrt(pn))) ** 2

    # (*We estimate the error due to the finite M using the precomputed table.*)
    # For the extra nucleus we again replace \[Eta] (\[Eta] - 1) with \[Eta] (\[Eta] - 1 + 2 \[Zeta])
    epsM = (
        eps_mt[n_p - 1, nM - 1]
        * eta
        * (eta - 1 + 2 * zeta + 2 * lam_zeta)
        / (2 * np.pi * Omega ** (1 / 3))
    )
    epsM += eps_mt[n_n - 1, nM - 1] * 2 * zeta * lam_zeta

    # First we estimate the error due to the finite precision of the \
    # nuclear positions.
    #   The following formula is from the formula for the error due to the \
    # nuclear positions in Theorem 4,
    # where we have used (64*(2^np - 1))*
    #   p for the sum over 1/ | \[Nu] | .
    #    First we estimate the number of bits to obtain an error that is \
    # some small fraction of the total error, then use that to compute the \
    # actual bound in the error for that number of bits
    nrf = (64 * (2**n_p - 1)) * p * eta * lam_zeta / Omega ** (1 / 3)
    nrf += (64 * (2**n_n - 1)) * pn * zeta * lam_zeta / Omega ** (1 / 3)
    nR = nbr + np.rint(np.log2(nrf / eps))

    lam_1 = max(
        lam_T + lam_U + lam_Un + lam_V + lam_Vn,
        lam_Un / pn + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / p,
    ) / (
        Peq0 * Peq1 * Peq3
    )  # (*See Eq. (127).*)
    lam_2 = max(
        lam_T + lam_U + lam_Un + lam_V + lam_Vn,
        lam_Un / pnmp + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / pamp,
    ) / (
        Peq0 * Peq1 * Peq3
    )  #  (*See Eq. (126).*)

    lam_tot_temp = max(lam_1, lam_2)
    # print(lam_T + lam_U + lam_Un + lam_V + lam_Vn)

    #  (*See Eq. (133).*)
    epsR = nrf / 2**nR
    nT = 10 + np.rint(np.log2(lam_tot_temp / eps))
    epsT = 5 * lam_tot_temp / 2 ** (nT)

    # The number of iterations of the phase measurement.
    # In the following the 1/(1 - 1/\[Eta]) is replaced according to the following reasoning.
    # Note that in the discussion below Eq. (119) this expression comes from
    # \[Eta]^2/(\[Eta] (\[Eta] - 1)) for comparing the cases with and without inequaility tests.
    # Here we need the ratio (\[Eta] + \[Zeta])^2/(\[Eta] (\[Eta] - 1 + 2 \[Zeta])) instead
    if eps > epsM + epsR + epsT:
        eps_ph = np.sqrt(eps**2 - (epsM + epsR + epsT) ** 2)
    else:
        eps_ph = 10 ** (-100)

    # # # (*See Eq. (127).*)
    # lam_1 = max(lam_T + lam_U + lam_Un + lam_V + lam_Vn, lam_Un / pn + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / p) / (Peq0*Peq1* Peq3) # (*See Eq. (127).*)
    # lam_2 = max(lam_T + lam_U + lam_Un + lam_V + lam_Vn, lam_Un / pnmp + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / pamp) / (Peq0*Peq1* Peq3)  #  (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2\[Lambda]\[Zeta]) replaced with P_s(3,8). This is because we are taking \[Eta]=\[Lambda]\[Zeta].*)

    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = np.ceil(np.pi * lam_1 / (2 * eps_ph))
    m2 = np.ceil(np.pi * lam_2 / (2 * eps_ph))

    # (*Steps for phase estimation with amplitude amplification.*)

    # (*The number of bits used for the equal state preparation for \
    # choosing between U and V. This is significantly changed when we \
    # include the extra nucleus, because we have the relative weight 2(\
    # \[Eta]+\[Zeta])\[Lambda]\[Zeta] for \[Lambda]_U and \[Eta](\[Eta]-1+2\
    # \[Zeta]) for \[Lambda]_V, without \[Eta] factoring out. We need to \
    # prepare an equal superposition over \
    # 2(\[Eta]+\[Zeta])\[Lambda]\[Zeta]+\[Eta](\[Eta]-1+2\[Zeta]) numbers \
    # because of this.*)
    n_eta_zeta = 0
    n_eta = np.ceil(np.log2(eta))

    # the c1 cost is replaced with the cost of inequality tests
    c1 = 6 * nT - 1

    # Here the + \[Zeta] accounts for the equal superposition including the extra nucleus
    factors = factorint(eta)
    bts = factors[min(list(sorted(factors.keys())))]

    if (eta + zeta) % 2 > 0:
        bts = 0

    # (*This is cost of superposition over i and j. See Eq. (62), or table line 2.*)
    c2 = 14 * n_eta + 8 * br - 36 - 12 * bts

    # (*Table line 3.*)
    c3 = 2 * (2 * n_n + 9) + 2 * (n_n - n_p) + 20

    # (*Table line 4.*)
    # this cost of controlled swaps
    # into and out of ancilla
    # need 6 * nn to acount for extra qubits for nucleus
    c4 = 12 * eta * n_p + 6 * n_n + 4 * eta - 6

    # (*Table line 5.*)
    # include extra cost for the nuclear momentum as well as 4 more Toff for selecting x component
    c5 = 5 * n_n - 2

    # (*Table line 6, modified?.*)
    # We need to account for the extra cost for the nuclear dimension.
    # The (nn - np) is extra costs for making the preparation of nested boxes \
    # controlled. The Toffolis can be undone with Cliffords
    c6 = 3 * n_n**2 + 16 * n_n - n_p - 6 + 4 * nM * (n_n + 1)

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)

    # this is for additions and subtractions of momenta, but only one of \
    # the registers can have the higher dimension for the nucleus.
    c8 = 12 * n_p + 12 * n_n

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)])
    c9 = 3 * (2 * n_n * nR - n_n * (n_n + 1) - 1 if nR > n_n else nR * (nR - 1))

    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = nT + 2 * n_eta + 6 * n_n + nM + 16

    # # (*The extra costs for accounting for the extra nucleus that is treated quantum mechanically.*)
    # cnuc = 4 * (nn - np) + 6 * nT - 1 + np - 1 + 2

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr) * m1
    # (*Next the cost with the amplitude amplification.*)
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3 * c6 + c7 + c8 + c9 + cr) * m2

    # (*Qubits storing the momenta. Here \[Eta] is replaced with \[Eta]+1 for the extra nucleus.*)
    # state qubits
    q1 = 3 * (eta * n_p + n_n)

    # Qubits for phase estimation
    q2 = 2 * np.ceil(np.log2(m1 if cq < cqaa else m2)) - 1

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 = nR + 1

    # (*The |T> state.*)
    q4 = 1
    # (*The rotated qubit for T vs U+V.*)  # in prep
    q5 = 1

    # (*The superposition state for selecting between U and V. This is changed from n\[Eta]\[Zeta]+3 to bL+4, with Log2[L] for outputting L.*)
    # in prep
    q6 = np.ceil(np.log2(L)) + 4

    # (*The result of a Toffoli on the last two.*)
    q7 = 1

    # (*Preparing the superposition over i and j.*)
    q8 = 2 * n_eta + 5

    # (*For preparing the superposition over \[Nu].*)
    q9 = (
        3 * (n_p + 1)
        + n_p
        + nM
        + (3 * n_p + 2)
        + (2 * n_p + 1)
        + (3 * n_p**2 - n_p - 1 + 4 * nM * (n_p + 1))
        + 1
        + 2
    )

    # (*The nuclear positions.*)
    q10 = 3 * nR

    # (*Preparation of w.*)
    q11 = 4

    # (*Preparation of w, r and s.*)
    q12 = 2 * n_p + 4
    # (*Temporary qubits for updating momenta.*)
    q13 = 5 * n_p + 1

    # (*Overflow bits for arithmetic on momenta.*)
    q14 = 6

    # (*Arithmetic for phasing for nuclear positions.*)
    q15 = 2 * (nR - 2)
    qt = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12 + q13 + q14

    # return block encoding cost and qubit requirement without phase estimation qubits
    if cq < cqaa:
        return cq / m1, lam_1, int(qt) - int(q2)
    else:
        return cqaa / m2, lam_2, int(qt) - int(q2)


@frozen
class Synthesizer(PrepareOracle):
    """Cost to build Sythesizer

    This synthesizer produces
    $$
    P\\vert0\\rangle = \\sum_{w \\in W} \\sqrt{p(w)} \\vert w\\rangle \\vert garbage_{w}\\rangle
    $$

    where $$|w\\rangle$$ is the selection register and $$|\\mathrm{garbage}_{w}\\rangle$$
    is the ancilla space associate with the walk operator
    """

    stopping_system: StoppingPowerSystem
    evolution_precision: float
    evolution_time: float

    @cached_property
    def _pw_qubitization_costs_from_v5(self) -> Tuple[float, float, int]:
        """Returns cost of PREPARE as a tuple of (num_toffoli, lambda, num_logical_qubits)"""
        return _pw_qubitization_with_projectile_costs_from_v5(self.stopping_system)

    @cached_property
    def selection_registers(self) -> cirq_infra.SelectionRegisters:
        num_sys_qubits = self.stopping_system.get_system_size()
        return cirq_infra.SelectionRegisters.build(selection=(num_sys_qubits, 2**num_sys_qubits))

    @cached_property
    def junk_registers(self) -> cirq_infra.Registers:
        _, _, num_logical = self._pw_qubitization_costs_from_v5
        return cirq_infra.Registers.build(junk=num_logical - self.selection_registers.bitsize)

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        # No decompose specified.
        return NotImplemented

    def _t_complexity_(self) -> cq.TComplexity:
        # Total toffoli times
        # 2(λt + 1.04(λt)⅓)log(1/ε)⅔
        block_encoding_toff, lambdaval, _ = self._pw_qubitization_costs_from_v5
        lambda_by_time = np.abs(self.evolution_time) * lambdaval
        num_queries_to_block_encoding = (
            2
            * (lambda_by_time + 1.04 * (lambda_by_time) ** (1 / 3))
            * np.log2(1 / self.evolution_precision) ** (2 / 3)
        )
        return cq.TComplexity(t=4 * num_queries_to_block_encoding * block_encoding_toff)
