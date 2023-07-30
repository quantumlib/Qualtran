"""THC SELECT and PREPARE"""
from functools import cached_property
from typing import Dict, Optional, Set, Tuple, Union

import sympy
from attrs import frozen

from qualtran import Bloq, BloqBuilder, Register, Signature, SoquetT
from qualtran.bloqs.arithmetic import LessThanEqual
from qualtran.bloqs.basic_gates import Hadamard, TGate
from qualtran.bloqs.chemistry import ContiguousRegister, MultiControl
from qualtran.bloqs.qrom import QROM
from qualtran.bloqs.swap_network import CSwapApprox
from qualtran.resource_counting import SympySymbolAllocator


@frozen
class UniformSuperpositionTHC(Bloq):
    r"""Prepare uniform superposition state for THC.

    $$
        |0\rangle^{\otimes 2\log(M+1)} \rightarrow \sum_{\mu\le\nu}^{M} |\mu\rangle|\nu\rangle + \sum_{\mu}^{N/2}|\mu\rangle|\nu=M+1\rangle,
    $$

    where $M$ is the THC auxiliary dimension, and $N$ is the number of spin orbitals.

    Args:
        bitsize: the number of bits needed for the mu and nu registers. Should
            be large enough to represent $M+1$.
        bitsize_rot: the number of bits needed for the Ry gate for amplitude
            amplification which is implemented using a phase gradient gate.

    Registers:
    - mu: $\mu$ register.
    - nu: $\nu$ register.
    - succ: ancilla flagging success of amplitude amplification.
    - junk: ancillas for inequality testing.
    - amp: ancilla for amplitude amplification.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf). Eq. 29.
    """

    bitsize: int
    bitsize_rot: int = 7

    @cached_property
    def signature(self) -> Signature:
        return Signature(
            [
                Register("mu", bitsize=self.bitsize),
                Register("nu", bitsize=self.bitsize),
                Register("succ", bitsize=1),
                # Register("junk", bitsize=1, shape=(5,)),
                # Register("amp", bitsize=1, shape=(self.bitsize_rot,)),
            ]
        )

    def bloq_counts(
        self, ssa: Optional['SympySymbolAllocator'] = None
    ) -> Set[Tuple[Union[int, sympy.Expr], Bloq]]:
        toff_count = 10 * self.bitsize + 2 * self.bitsize_rot - 9
        return {(4 * toff_count, TGate())}


@frozen
class PrepareTHC(Bloq):
    r"""State Preparation for THC Hamilontian.

    Prepares the state

    $$
        \frac{1}{\sqrt{\lambda}}|+\rangle|+\rangle\left[
            \sum_\ell^{N/2} \sqrt{t_\ell}|\ell\rangle|M+1\rangle
            + \frac{1}{\sqrt{2}} \sum_{\mu\le\nu}^M \sqrt{\zeta_{\mu\nu}} |\mu\rangle|\nu\rangle
        \right].
    $$

    Args:
        num_mu: THC auxiliary index dimension $M$
        num_spin_orb: number of spin orbitals $N$
        keep_bitsize: number of bits for keep register for coherent alias sampling.

    Registers:
     - mu: $\mu$ register.
     - nu: $\nu$ register.
     - theta: sign register.
     - succ: ancilla flagging success of amplitude amplification.
     - alt_mu: Register for alt_mu values.
     - alt_nu: Register for alt_nu values.
     - keep: Register for keep values.
     - flag_plus: ancilla for creating $|+\rangle$ state.
     - nu_eq_M_pl_1: ancilla for flagging $\nu = M + 1$.

    References:
        [Even more efficient quantum computations of chemistry through
            tensor hypercontraction](https://arxiv.org/pdf/2011.03494.pdf) Fig. 2 and Fig. 3.
    """

    num_mu: int
    num_spin_orb: int
    keep_bitsize: int
    adjoint: bool = False

    @cached_property
    def signature(self) -> Signature:
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        return Signature(
            [
                Register("mu", bitsize=(self.num_mu - 1).bit_length()),
                Register("nu", bitsize=(self.num_mu - 1).bit_length()),
                Register("theta", bitsize=1),
                Register("succ", bitsize=1),
                Register("s", bitsize=(data_size - 1).bit_length()),
                Register("alt_theta", bitsize=1),
                Register("alt_mu", bitsize=(self.num_mu - 1).bit_length()),
                Register("alt_nu", bitsize=(self.num_mu - 1).bit_length()),
                Register("keep", bitsize=self.keep_bitsize),
                Register("sigma", bitsize=self.keep_bitsize),
                Register("less_than", bitsize=1),
                Register("flag_plus", bitsize=1),
                Register("nu_eq_M_pl_1", bitsize=1),
            ]
        )

    def build_composite_bloq(self, bb: 'BloqBuilder', **regs: SoquetT) -> Dict[str, 'SoquetT']:
        data_size = self.num_spin_orb // 2 + self.num_mu * (self.num_mu + 1) // 2
        log_mu = (self.num_mu - 1).bit_length()
        log_d = (data_size - 1).bit_length()
        # 1. Prepare THC uniform superposition over mu, nu and succ
        mu, nu, succ = bb.add(
            UniformSuperpositionTHC(bitsize=log_mu), mu=regs['mu'], nu=regs['nu'], succ=regs['succ']
        )
        # 2. Make contiguous register from mu and nu and store in register `s`.
        for k, v in regs.items():
            print(k, v)
        mu, nu, s = bb.add(ContiguousRegister(log_mu, log_d), mu=mu, nu=nu, s=regs['s'])
        # qrom_regs: Dict[str, 'SoquetT'] = {}
        qrom = QROM(
            data_bitsizes=(1, log_mu, log_mu, self.keep_bitsize),
            selection_bitsizes=(log_d,),
            data_ranges=(data_size,) * 4,
        )
        s, theta, alt_mu, alt_nu, keep = bb.add(
            qrom,
            selection0=s,
            target0=regs['theta'],
            target1=regs['alt_mu'],
            target2=regs['alt_nu'],
            target3=regs['keep'],
        )
        keep, sigma, less_than = bb.add(
            LessThanEqual(self.keep_bitsize, self.keep_bitsize),
            x=keep,
            y=regs['sigma'],
            z=regs['less_than'],
        )
        theta, less_than = bb.add(MultiControl((1, 1), (1, 0)), ctrl_0=theta, ctrl_1=less_than)
        less_than, alt_mu, mu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_mu, y=mu)
        less_than, alt_nu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=less_than, x=alt_nu, y=nu)
        keep, sigma, less_than = bb.add(
            LessThanEqual(self.keep_bitsize, self.keep_bitsize), x=keep, y=sigma, z=less_than
        )
        flag_plus = bb.add(Hadamard(), q=regs['flag_plus'])
        flag_plus, nu_eq_m_pl_one = bb.add(
            MultiControl(bitsizes=(1, 1), cvs=(1, 0)), ctrl_0=flag_plus, ctrl_1=regs['nu_eq_M_pl_1']
        )
        flag_plus, mu, nu = bb.add(CSwapApprox(bitsize=log_mu), ctrl=flag_plus, x=mu, y=nu)
        regs['mu'] = mu
        regs['nu'] = nu
        regs['s'] = s
        regs['succ'] = succ
        regs['theta'] = theta
        regs['alt_mu'] = alt_mu
        regs['alt_nu'] = alt_nu
        regs['keep'] = keep
        regs['sigma'] = sigma
        regs['less_than'] = less_than
        regs['flag_plus'] = flag_plus
        regs['nu_eq_M_pl_1'] = nu_eq_m_pl_one
        return regs
