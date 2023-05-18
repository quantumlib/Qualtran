from dataclasses import dataclass
from math import factorial
from typing import Sequence

import cirq
import numpy
from sympy import factorint

import cirq_qubitization as cq
from cirq_qubitization import cirq_infra
from cirq_qubitization.cirq_algos.mean_estimation.pvec_epsmat import epsmat as eps_mt
from cirq_qubitization.cirq_algos.mean_estimation.pvec_epsmat import pvec as pv


def M1(k, K):
    return numpy.ceil(
        numpy.log2(factorial(K) * numpy.sum([1 / factorial(k1) for k1 in range(k, K + 1)]))
    )


def g1(x, n):
    """g1[x_, n_] := Floor[2^n*ArcSin[(1/2)/Sqrt[x]]/(2 Pi)] * 2 Pi / 2^n"""
    asin_val = numpy.arcsin(0.5 / numpy.sqrt(x))
    floored_val = numpy.floor(2**n * asin_val / (2 * numpy.pi))
    return floored_val * 2 * numpy.pi / 2**n


def h1(x, n):
    return x * (
        (1 + (2 - 4 * x) * numpy.sin(g1(x, n)) ** 2) ** 2
        + 4 * numpy.sin(g1(x, n)) ** 2 * numpy.cos(g1(x, n)) ** 2
    )


def g2(x, n):
    asin_val = numpy.arcsin(0.5 / numpy.sqrt(x))
    return numpy.ceil(2**n * asin_val / (2 * numpy.pi)) * 2 * numpy.pi / 2**n


def h2(x, n):
    return x * (
        (1 + (2 - 4 * x) * numpy.sin(g2(x, n)) ** 2) ** 2
        + 4 * numpy.sin(g2(x, n)) ** 2 * numpy.cos(g2(x, n)) ** 2
    )


def h(x, n):
    return numpy.max([h1(x, n), h2(x, n)])


def Eq(n, br):
    return h(n / 2 ** (numpy.ceil(numpy.log2(n))), br)


def Er(zeta):
    kt1 = 2 ** numpy.floor(numpy.log2(zeta) / 2)
    kt2 = 2 ** numpy.ceil(numpy.log2(zeta) / 2)
    return numpy.min([numpy.ceil(zeta / kt1) + kt1, numpy.ceil(zeta / kt2) + kt2])


def _pw_qubitization_with_projectile_costs_from_v5(
    np: int,
    nn: int,
    eta: int,
    Omega: int,
    eps: float,
    nMc: int,
    nbr: int,
    L: int,
    zeta: float,
    mpr: float,
    kmean: float,
):
    """Internal function for costing out time evolution operator
    :params:
       np is the number of bits in each direction for the momenta
       nn is the number of bits in each direction for the nucleus
       eta is the number of electrons
       Omega cell volume in Bohr^3
       eps is the total allowable error
       nMc is an adjustment for the number of bits for M (used in nu preparation
       nbr is an adjustment in the number of bits used for the nuclear positions
       L is the number of nuclei
       zeta is the charge of the projectile
       mpr is the mass of the extra nucleus
       kmean is the mean momentum for the extra nucleas
    """
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
    p = pv[np - 1, 49]
    pn = pv[nn - 1, 49]

    # (*Now compute the lambda-values.*)
    # (*Here 64*(2^np-1))*p is \[Lambda]_\[Nu].*)
    tmp = (64 * (2**np - 1)) * p / (2 * numpy.pi * Omega ** (1 / 3))
    tmpn = (64 * (2**nn - 1)) * pn / (2 * numpy.pi * Omega ** (1 / 3))  # same but for nucleus

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
    # lam_T = 6 * (eta + zeta) * numpy.pi**2 / Omega**(2/3) * (2**(np - 1))**2
    lam_T = 6 * (eta + 1.0 / mpr) * numpy.pi**2 / Omega ** (2 / 3) * (
        2 ** (np - 1)
    ) ** 2 + 2 * numpy.pi * kmean / (mpr * Omega ** (1 / 3)) * (2 ** (nn - 1)) ** 2 / (
        2 ** (nn - 1) - 1
    )

    # (*Adjust value of nM based on \[Lambda]UV we just calculated.*)
    nM = nMc + int(numpy.rint(numpy.log2(20 * lam_UV / eps)))

    #  (*Recompute p and \[Lambda]V.*)
    p = pv[np - 1, nM - 1]
    pn = pv[nn - 1, nM - 1]
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
    pamp = numpy.sin(3 * numpy.arcsin(numpy.sqrt(p))) ** 2
    pnmp = numpy.sin(3 * numpy.arcsin(numpy.sqrt(pn))) ** 2

    # (*We estimate the error due to the finite M using the precomputed table.*)
    # For the extra nucleus we again replace \[Eta] (\[Eta] - 1) with \[Eta] (\[Eta] - 1 + 2 \[Zeta])
    epsM = (
        eps_mt[np - 1, nM - 1]
        * eta
        * (eta - 1 + 2 * zeta + 2 * lam_zeta)
        / (2 * numpy.pi * Omega ** (1 / 3))
    )
    epsM += eps_mt[nn - 1, nM - 1] * 2 * zeta * lam_zeta

    # First we estimate the error due to the finite precision of the \
    # nuclear positions.
    #   The following formula is from the formula for the error due to the \
    # nuclear positions in Theorem 4,
    # where we have used (64*(2^np - 1))*
    #   p for the sum over 1/ | \[Nu] | .
    #    First we estimate the number of bits to obtain an error that is \
    # some small fraction of the total error, then use that to compute the \
    # actual bound in the error for that number of bits
    nrf = (64 * (2**np - 1)) * p * eta * lam_zeta / Omega ** (1 / 3)
    nrf += (64 * (2**nn - 1)) * pn * zeta * lam_zeta / Omega ** (1 / 3)
    nR = nbr + numpy.rint(numpy.log2(nrf / eps))

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
    nT = 10 + numpy.rint(numpy.log2(lam_tot_temp / eps))
    epsT = 5 * lam_tot_temp / 2 ** (nT)

    # The number of iterations of the phase measurement.
    # In the following the 1/(1 - 1/\[Eta]) is replaced according to the following reasoning.
    # Note that in the discussion below Eq. (119) this expression comes from
    # \[Eta]^2/(\[Eta] (\[Eta] - 1)) for comparing the cases with and without inequaility tests.
    # Here we need the ratio (\[Eta] + \[Zeta])^2/(\[Eta] (\[Eta] - 1 + 2 \[Zeta])) instead
    if eps > epsM + epsR + epsT:
        eps_ph = numpy.sqrt(eps**2 - (epsM + epsR + epsT) ** 2)
    else:
        eps_ph = 10 ** (-100)

    # # # (*See Eq. (127).*)
    # lam_1 = max(lam_T + lam_U + lam_Un + lam_V + lam_Vn, lam_Un / pn + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / p) / (Peq0*Peq1* Peq3) # (*See Eq. (127).*)
    # lam_2 = max(lam_T + lam_U + lam_Un + lam_V + lam_Vn, lam_Un / pnmp + (lam_U + lam_Vn + lam_V / (1 - 1 / eta)) / pamp) / (Peq0*Peq1* Peq3)  #  (*See Eq. (126).*)
    # (*The P_eq is from Eq. (130), with P_s(\[Eta]+2\[Lambda]\[Zeta]) replaced with P_s(3,8). This is because we are taking \[Eta]=\[Lambda]\[Zeta].*)

    #  (*Steps for phase estimation without amplitude amplification.*)
    m1 = numpy.ceil(numpy.pi * lam_1 / (2 * eps_ph))
    m2 = numpy.ceil(numpy.pi * lam_2 / (2 * eps_ph))

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
    n_eta = numpy.ceil(numpy.log2(eta))

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
    c3 = 2 * (2 * nn + 9) + 2 * (nn - np) + 20

    # (*Table line 4.*)
    # this cost of controlled swaps
    # into and out of ancilla
    # need 6 * nn to acount for extra qubits for nucleus
    c4 = 12 * eta * np + 6 * nn + 4 * eta - 6

    # (*Table line 5.*)
    # include extra cost for the nuclear momentum as well as 4 more Toff for selecting x component
    c5 = 5 * nn - 2

    # (*Table line 6, modified?.*)
    # We need to account for the extra cost for the nuclear dimension.
    # The (nn - np) is extra costs for making the preparation of nested boxes \
    # controlled. The Toffolis can be undone with Cliffords
    c6 = 3 * nn**2 + 16 * nn - np - 6 + 4 * nM * (nn + 1)

    # (*The QROM cost according to the number of nuclei, line 7 modified.*)
    c7 = L + Er(L)

    # this is for additions and subtractions of momenta, but only one of \
    # the registers can have the higher dimension for the nucleus.
    c8 = 12 * np + 12 * nn

    # c9 = 3*(Piecewise[{{2*np*nR - np*(np + 1) - 1, nR > np}}, nR*(nR - 1)])
    c9 = 3 * (2 * nn * nR - nn * (nn + 1) - 1 if nR > nn else nR * (nR - 1))

    # (*The number of qubits we are reflecting on according to equation (136).*)
    cr = nT + 2 * n_eta + 6 * nn + nM + 16

    # # (*The extra costs for accounting for the extra nucleus that is treated quantum mechanically.*)
    # cnuc = 4 * (nn - np) + 6 * nT - 1 + np - 1 + 2

    # (*First the cost without the amplitude amplification.*)
    cq = (c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + cr) * m1
    # (*Next the cost with the amplitude amplification.*)
    cqaa = (c1 + c2 + c3 + c4 + c5 + 3 * c6 + c7 + c8 + c9 + cr) * m2

    # (*Qubits storing the momenta. Here \[Eta] is replaced with \[Eta]+1 for the extra nucleus.*)
    # state qubits
    q1 = 3 * (eta * np + nn)

    # Qubits for phase estimation
    q2 = 2 * numpy.ceil(numpy.log2(m1 if cq < cqaa else m2)) - 1

    # (*We are costing WITH nuclei, so the maximum precision of rotations is nR+1.*)
    q3 = nR + 1

    # (*The |T> state.*)
    q4 = 1
    # (*The rotated qubit for T vs U+V.*)  # in prep
    q5 = 1

    # (*The superposition state for selecting between U and V. This is changed from n\[Eta]\[Zeta]+3 to bL+4, with Log2[L] for outputting L.*)
    # in prep
    q6 = numpy.ceil(numpy.log2(L)) + 4

    # (*The result of a Toffoli on the last two.*)
    q7 = 1

    # (*Preparing the superposition over i and j.*)
    q8 = 2 * n_eta + 5

    # (*For preparing the superposition over \[Nu].*)
    q9 = (
        3 * (np + 1)
        + np
        + nM
        + (3 * np + 2)
        + (2 * np + 1)
        + (3 * np**2 - np - 1 + 4 * nM * (np + 1))
        + 1
        + 2
    )

    # (*The nuclear positions.*)
    q10 = 3 * nR

    # (*Preparation of w.*)
    q11 = 4

    # (*Preparation of w, r and s.*)
    q12 = 2 * np + 4
    # (*Temporary qubits for updating momenta.*)
    q13 = 5 * np + 1

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


@dataclass
class StoppingPowerSystem:
    """Data representing the system we are stopping and projectile
    :params:
        np is the number of bits in each direction for the momenta
        nn is the number of bits in each direction for the nucleus
        eta is the number of electrons
        Omega cell volume in Bohr^3
        eps is the total allowable error
        nMc is an adjustment for the number of bits for M (used in nu preparation
        nbr is an adjustment in the number of bits used for the nuclear positions
        L is the number of nuclei
        zeta is the charge of the projectile
        mpr is the mass of the extra nucleus
        kmean is the mean momentum for the extra nucleas
    """

    np: int
    nn: int
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
        return 3 * (self.eta * self.np + self.nn)

    def get_kmean_in_au(self):
        projectile_ke = 0.5 * self.mpr * self.kmean**2
        return numpy.sqrt(2 * projectile_ke / self.mpr) * self.mpr  # p = m * v


class SynthesizeOracle(cirq_infra.GateWithRegisters):
    """Cost to build sythesizer

    This synthesizer produces
    $$
    P\\vert0\\rangle = \\sum_{w \\in W} \\sqrt{p(w)} \\vert w\\rangle \\vert garbage_{w}\\rangle
    $$

    where $$|w\\rangle$$ is the system register and $$|\\mathrm{garbage}_{w}\\rangle$$
    is the ancilla space associate with the walk operator
    """

    def __init__(
        self,
        stopping_system: StoppingPowerSystem,
        evolution_precision: float,
        evolution_time: float,
    ):
        self.stop_sys = stopping_system
        self.evolution_precision = evolution_precision
        self.tau = evolution_time

        (
            blockencodingtoff,
            lambdaval,
            numlogicalqubit,
        ) = _pw_qubitization_with_projectile_costs_from_v5(
            np=self.stop_sys.np,
            nn=self.stop_sys.nn,
            eta=self.stop_sys.eta,
            Omega=self.stop_sys.Omega,
            eps=self.stop_sys.eps,
            nMc=self.stop_sys.nMc,
            nbr=self.stop_sys.nbr,
            L=self.stop_sys.L,
            zeta=self.stop_sys.zeta,
            mpr=self.stop_sys.mpr,
            kmean=self.stop_sys.get_kmean_in_au(),
        )
        self.block_encoding_toff = blockencodingtoff
        self.lambda_val = lambdaval
        self.num_sys_qubits = self.stop_sys.get_system_size()
        self.num_garbage_qubits = numlogicalqubit - self.num_sys_qubits

        # Total toffoli times
        # 2(λt + 1.04(λt)⅓)log(1/ε)⅔
        lambda_by_time = numpy.abs(self.tau) * lambdaval
        self.num_queries_to_block_encoding = (
            2
            * (lambda_by_time + 1.04 * (lambda_by_time) ** (1 / 3))
            * numpy.log2(1 / self.evolution_precision) ** (2 / 3)
        )

    def system_register(self) -> cirq_infra.Register:
        return cirq_infra.Register('system', self.num_sys_qubits)

    def garbage_register(self) -> cirq_infra.Register:
        return cirq_infra.Register('garbage', self.num_garbage_qubits)

    def registers(self) -> cirq_infra.Registers:
        return cirq_infra.Registers([self.system_register, self.garbage_register])

    def _num_qubits_(self) -> int:
        return self.registers.bitsize

    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        return super().decompose_from_registers(**qubit_regs)

    def _t_complexity_(self) -> cq.TComplexity:
        return cq.TComplexity(t=4 * self.num_queries_to_block_encoding * self.block_encoding_toff)


if __name__ == "__main__":
    ########################################################
    #
    # I will formalize this as a test before official merge
    #
    ########################################################
    Omega = 2419.682821036357
    num_bits_momenta = 6  # Number of bits in each direction for momenta
    num_bits_nuclei_momenta = 8
    eps_total = 1e-3  # Total allowable error
    num_bits_nu = 6  # extra bits for nu
    num_bits_nuc = 8  # extra bits for nuclear positions
    num_nuclei = 216  # L
    num_elec = 218  # eta
    projectile_mass = 1836 * 4
    projectile_charge = 2  # Helium
    projectile_velocity = 4
    nbr = 20

    sts = StoppingPowerSystem(
        np=num_bits_momenta,
        nn=num_bits_nuclei_momenta,
        eta=num_elec,
        Omega=Omega,
        eps=eps_total,
        nMc=num_bits_nuc,
        nbr=nbr,
        L=num_nuclei,
        zeta=projectile_charge,
        mpr=projectile_mass,
        kmean=projectile_velocity,
    )

    so_inst = SynthesizeOracle(stopping_system=sts, evolution_precision=1.0e-3, evolution_time=2)
    print(so_inst)
