from functools import cached_property
from typing import Dict, Tuple

import attrs
import numpy as np
import sympy
from attrs import frozen
from cirq_ft.infra.t_complexity_protocol import TComplexity

from cirq_qubitization.bloq_algos.basic_gates.cnot import CNOT
from cirq_qubitization.bloq_algos.basic_gates.hadamard import HGate
from cirq_qubitization.bloq_algos.basic_gates.t_gate import TGate
from cirq_qubitization.bloq_algos.basic_gates.toffoli import ToffoliGate
from cirq_qubitization.bloq_algos.basic_gates.x_basis import XGate
from cirq_qubitization.bloq_algos.basic_gates.z_basis import ZGate
from cirq_qubitization.bloq_algos.swap_network import CSwapApprox
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.bloq_counts import big_O
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloqBuilder, SoquetT
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters


@frozen
class SelectedMajoranaFermion(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    selection_bitsizes: Tuple[int, ...]
    target_bitsize: int
    gate: Bloq
    cvs: Tuple[int, ...] = tuple()

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [
            FancyRegister(f's_{i}', bitsize=bs) for (i, bs) in enumerate(self.selection_bitsizes)
        ]
        regs += [FancyRegister('t', bitsize=self.target_bitsize)]
        regs += [FancyRegister(f'c_{i}', bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    # def pretty_name(self) -> str:
    #     name = f"{self.gate}"[0]
    #     return r'In[ZX/Y]'

    def t_complexity(self) -> TComplexity:
        iteration_size = sum([np.prod(2**it_shape) for it_shape in self.selection_bitsizes])
        return TComplexity(t=4 * iteration_size - 4)


@frozen
class QROM(Bloq):
    data_bitsizes: Tuple[int, ...]
    selection_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...]

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"s_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)]
        regs = [FancyRegister(f"t_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        return r'QROM'

    def t_complexity(self) -> TComplexity:
        t_count = 4 * np.prod([sympy.exp(s) for s in self.selection_bitsizes])
        return TComplexity(t=big_O(t_count))

    def rough_decompose(self, mgr):
        t_count = big_O(4 * np.prod(sympy.exp(s) for s in self.selection_bitsizes))
        return [(t_count, TGate())]


@frozen
class QROAM(Bloq):
    selection_bitsizes: Tuple[int, ...]
    data_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"s_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)]
        regs += [FancyRegister(f"t_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        return r'QROAM'

    def t_complexity(self) -> TComplexity:
        t_count = 4 * np.prod(sympy.exp(s) for s in self.selection_bitsizes)
        return TComplexity(t=big_O(t_count))

    def rough_decompose(self, mgr):
        if isinstance(self.selection_bitsizes[0], sympy.Symbol):
            t_count = big_O(np.prod(sympy.exp(s) for s in self.selection_bitsizes) ** 0.5)
        else:
            t_count = int(4 * np.prod([2 ** (s) for s in self.selection_bitsizes]) ** 0.5)
        return [(t_count, TGate())]


@frozen
class Prepare(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
        (Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity)[https://arxiv.org/abs/1805.03662].
            Babbush et. al. 2018. Section III.A. and Fig. 4.
    """

    data_bitsizes: Tuple[int, ...]
    output_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        regs = [FancyRegister(f"d_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"o_{i}", bitsize=bs) for i, bs in enumerate(self.output_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'Prep{dag}'

    @staticmethod
    def _map_names(regs, out_dict, in_str, out_str):
        indx = 0
        for k, v in regs.items():
            if in_str in k:
                out_dict[f'{out_str}_{indx}'] = v
                indx += 1

    @staticmethod
    def _update_regs(new_regs, out_regs, in_str, out_str):
        indx = 0
        for k, v in new_regs.items():
            if in_str in k:
                out_key = f'{out_str}_{indx}'
                indx += 1
                out_regs[out_key] = new_regs[k]

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        qroam_regs: Dict[str, 'SoquetT'] = {}
        self._map_names(regs, qroam_regs, 'd', 's')
        self._map_names(regs, qroam_regs, 'o', 't')
        self._map_names(regs, qroam_regs, 'c', 'c')
        qr = QROAM(
            selection_bitsizes=self.data_bitsizes, data_bitsizes=self.output_bitsizes, cvs=self.cvs
        )
        out = bb.add(qr, **qroam_regs)
        new_regs = {k: v for k, v in zip(qroam_regs.keys(), out)}
        self._update_regs(new_regs, regs, 's', 'd')
        self._update_regs(new_regs, regs, 't', 'o')
        self._update_regs(new_regs, regs, 'c', 'c')
        return regs


@frozen
class IndexedPrepare(Bloq):
    """SelectMajoranaFermion Bloq
    Args:

    Registers:

    References:
    """

    selection_bitsizes: Tuple[int, ...]
    data_bitsizes: Tuple[int, ...]
    output_bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...] = tuple()
    adjoint: bool = False

    @cached_property
    def registers(self) -> FancyRegisters:
        # ignoring junk registers for the moment (alt/keep/...)
        regs = [FancyRegister(f"s_{i}", bitsize=bs) for i, bs in enumerate(self.selection_bitsizes)]
        regs += [FancyRegister(f"d_{i}", bitsize=bs) for i, bs in enumerate(self.data_bitsizes)]
        regs += [FancyRegister(f"o_{i}", bitsize=bs) for i, bs in enumerate(self.output_bitsizes)]
        regs += [FancyRegister(f"c_{i}", bitsize=1) for i in range(len(self.cvs))]
        return FancyRegisters(regs)

    def pretty_name(self) -> str:
        dag = '†' if self.adjoint else ''
        return rf'IndPrep{dag}'

    @staticmethod
    def _map_names(regs, out_dict, in_str, out_str):
        indx = 0
        for k, v in regs.items():
            if in_str in k:
                out_dict[f'{out_str}_{indx}'] = v
                indx += 1

    @staticmethod
    def _update_regs(new_regs, out_regs, in_str, out_str):
        indx = 0
        for k, v in new_regs.items():
            if in_str in k:
                out_key = f'{out_str}_{indx}'
                indx += 1
                out_regs[out_key] = new_regs[k]

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        qroam_regs: Dict[str, 'SoquetT'] = {}
        self._map_names(regs, qroam_regs, 's', 's')
        self._map_names(regs, qroam_regs, 'o', 't')
        self._map_names(regs, qroam_regs, 'c', 'c')
        qr = QROAM(
            selection_bitsizes=self.selection_bitsizes,
            data_bitsizes=self.output_bitsizes,
            cvs=self.cvs,
        )
        out = bb.add(qr, **qroam_regs)
        new_regs = {k: v for k, v in zip(qroam_regs.keys(), out)}
        self._update_regs(new_regs, regs, 's', 's')
        self._update_regs(new_regs, regs, 't', 'o')
        self._update_regs(new_regs, regs, 'c', 'c')
        return regs


@frozen
class AddMod(Bloq):
    input_bitsize: int
    output_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('input', self.input_bitsize),
                FancyRegister('output', self.output_bitsize),
            ]
        )

    def rough_decompose(self, mgr):
        if isinstance(self.input_bitsize, sympy.Symbol):
            # t_count = big_O(4 * np.prod(self.iteration_ranges) - 4
            t_count = big_O(self.input_bitsize)
        else:
            # TODO: this is worst case use proper decompose
            t_count = 2 * self.input_bitsize
        return [(t_count, TGate())]


@frozen
class MultiControlledPauli(Bloq):
    bitsizes: Tuple[int, ...]
    gate: Bloq

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters([FancyRegister(f'c_{i}', bs) for i, bs in enumerate(self.bitsizes)])

    # def pretty_name(self) -> str:
    #     return f'C^n{self.gate}'[:-6]

    def rough_decompose(self, mgr):
        if isinstance(self.bitsizes, sympy.Symbol):
            t_count = big_O(4 * sum(self.bitsizes) - 8)
        else:
            t_count = 4 * sum(self.bitsizes) - 8
        return [(t_count, TGate())]


@frozen
class SingleFactorization(Bloq):
    l_bitsize: int
    k_bitsize: int
    p_bitsize: int
    target_bitsize: int

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('succ', 1),
                FancyRegister('l', self.l_bitsize),
                FancyRegister('two_body', 1),
                FancyRegister('Q', self.k_bitsize),  # TODO fix wireshape
                FancyRegister('succ_kpq', 1),
                FancyRegister('k', self.k_bitsize),  # TODO fix wireshape
                FancyRegister('p', self.p_bitsize),
                FancyRegister('q', self.p_bitsize),
                FancyRegister('ReIm', 1),
                FancyRegister('AB', 1),
                FancyRegister('term', 1),
                FancyRegister('anc', 1),
                FancyRegister('alpha', 1),
                FancyRegister('psia', self.target_bitsize),
                FancyRegister('psib', self.target_bitsize),
            ]
        )

    def canon(self, bloq):
        if isinstance(bloq, Prepare):
            return attrs.evolve(
                bloq,
                data_bitsizes=(self.l_bitsize,),
                output_bitsizes=(1, self.k_bitsize),
                cvs=(1,),
                adjoint=False,
            )
        if isinstance(bloq, IndexedPrepare):
            return attrs.evolve(
                bloq,
                selection_bitsizes=(
                    self.l_bitsize,
                    self.k_bitsize,
                    self.p_bitsize,
                    self.p_bitsize,
                    1,
                ),
                adjoint=False,
            )

        return bloq

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', **regs: SoquetT
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls-1.

        This method builds the `adjoint=False` composite bloq. `self.decompose_bloq()`
        will throw if `self.adjoint=True`.
        """
        # 1. Prepare
        out = {}
        l, two_body, Q, succ = bb.add(
            Prepare(data_bitsizes=(self.l_bitsize,), output_bitsizes=(1, self.k_bitsize), cvs=(1,)),
            d_0=regs['l'],
            o_0=regs['two_body'],
            o_1=regs['Q'],
            c_0=regs['succ'],
        )
        l, k, p, q, re_im, succ_kpq = bb.add(
            # out = bb.add(
            IndexedPrepare(
                selection_bitsizes=(
                    self.l_bitsize,
                    self.k_bitsize,
                    self.p_bitsize,
                    self.p_bitsize,
                    1,
                ),
                # data_bitsizes=(alt_bitsize+keep_bitsize+1,)
                data_bitsizes=(),
                output_bitsizes=(),
                cvs=(1,),
            ),
            s_0=l,
            s_1=regs['k'],
            s_2=regs['p'],
            s_3=regs['q'],
            s_4=regs['ReIm'],
            c_0=regs['succ_kpq'],
        )
        (ab,) = bb.add(HGate(), q=regs['AB'])
        (term,) = bb.add(HGate(), q=regs['term'])
        (anc,) = bb.add(HGate(), q=regs['anc'])
        (alpha,) = bb.add(HGate(), q=regs['alpha'])
        (alpha, psia, psib) = bb.add(
            CSwapApprox(self.target_bitsize), ctrl=alpha, x=regs['psia'], y=regs['psib']
        )
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        (ab, two_body, re_im) = bb.add(ToffoliGate(), c0=ab, c1=two_body, t=re_im)
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (Q, q, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=Q, s_1=q, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        # # Missing Multi control
        succ, succ_kpq, re_im, term = bb.add(
            MultiControlledPauli((1, 1, 1, 1), ZGate()), c_0=succ, c_1=succ_kpq, c_2=re_im, c_3=term
        )
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (k, p, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=k, s_1=p, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        (ab, two_body, re_im) = bb.add(ToffoliGate(), c0=ab, c1=two_body, t=re_im)
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (alpha, psia, psib) = bb.add(CSwapApprox(self.target_bitsize), ctrl=alpha, x=psia, y=psib)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        l, k, p, q, re_im, succ_kpq = bb.add(
            IndexedPrepare(
                selection_bitsizes=(
                    self.l_bitsize,
                    self.k_bitsize,
                    self.p_bitsize,
                    self.p_bitsize,
                    1,
                ),
                # data_bitsizes=(alt_bitsize+keep_bitsize+1,)
                data_bitsizes=(),
                output_bitsizes=(),
                cvs=(1,),
                adjoint=True,
            ),
            s_0=l,
            s_1=k,
            s_2=p,
            s_3=q,
            s_4=re_im,
            c_0=succ_kpq,
        )
        (ab,) = bb.add(HGate(), q=ab)
        (term,) = bb.add(HGate(), q=term)
        (anc,) = bb.add(HGate(), q=anc)
        (alpha,) = bb.add(HGate(), q=alpha)
        succ, two_body, k, p, q, re_im, ab, term, anc, alpha = bb.add(
            MultiControlledPauli(
                (1, 1, self.k_bitsize, self.p_bitsize, self.p_bitsize, 1, 1, 1, 1, 1), ZGate()
            ),
            c_0=succ,
            c_1=two_body,
            c_2=k,
            c_3=p,
            c_4=q,
            c_5=re_im,
            c_6=ab,
            c_7=term,
            c_8=anc,
            c_9=alpha,
        )
        # Second half
        l, k, p, q, re_im, succ_kpq = bb.add(
            # out = bb.add(
            IndexedPrepare(
                selection_bitsizes=(
                    self.l_bitsize,
                    self.k_bitsize,
                    self.p_bitsize,
                    self.p_bitsize,
                    1,
                ),
                # data_bitsizes=(alt_bitsize+keep_bitsize+1,)
                data_bitsizes=(),
                output_bitsizes=(),
                cvs=(1,),
            ),
            s_0=l,
            s_1=k,
            s_2=p,
            s_3=q,
            s_4=re_im,
            c_0=succ_kpq,
        )
        (ab,) = bb.add(HGate(), q=ab)
        (term,) = bb.add(HGate(), q=term)
        (anc,) = bb.add(HGate(), q=anc)
        (alpha,) = bb.add(HGate(), q=alpha)
        (alpha, psia, psib) = bb.add(CSwapApprox(self.target_bitsize), ctrl=alpha, x=psia, y=psib)
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (Q, q, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=Q, s_1=q, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        succ, succ_kpq, re_im, term = bb.add(
            MultiControlledPauli((1, 1, 1, 1), ZGate()), c_0=succ, c_1=succ_kpq, c_2=re_im, c_3=term
        )
        (re_im, term) = bb.add(CNOT(), ctrl=re_im, target=term)
        smf = SelectedMajoranaFermion(
            selection_bitsizes=(self.k_bitsize, self.p_bitsize, 1),
            target_bitsize=self.target_bitsize,
            gate=XGate(),
            cvs=(1, 1),
        )
        (k, p, term, psia, succ, succ_kpq) = bb.add(
            smf, s_0=k, s_1=p, s_2=term, t=psia, c_0=succ, c_1=succ_kpq
        )
        k, Q = bb.add(AddMod(self.k_bitsize, self.k_bitsize), input=k, output=Q)
        (alpha, psia, psib) = bb.add(CSwapApprox(self.target_bitsize), ctrl=alpha, x=psia, y=psib)
        (anc, p, q) = bb.add(CSwapApprox(self.p_bitsize), ctrl=anc, x=p, y=q)
        l, k, p, q, re_im, succ_kpq = bb.add(
            IndexedPrepare(
                selection_bitsizes=(
                    self.l_bitsize,
                    self.k_bitsize,
                    self.p_bitsize,
                    self.p_bitsize,
                    1,
                ),
                # data_bitsizes=(alt_bitsize+keep_bitsize+1,)
                data_bitsizes=(),
                output_bitsizes=(),
                cvs=(1,),
                adjoint=True,
            ),
            s_0=l,
            s_1=k,
            s_2=p,
            s_3=q,
            s_4=re_im,
            c_0=succ_kpq,
        )
        (ab,) = bb.add(HGate(), q=ab)
        (term,) = bb.add(HGate(), q=term)
        (anc,) = bb.add(HGate(), q=anc)
        (alpha,) = bb.add(HGate(), q=alpha)
        l, two_body, Q, succ = bb.add(
            Prepare((self.l_bitsize,), (1, self.k_bitsize), cvs=(1,)),
            d_0=l,
            o_0=two_body,
            o_1=Q,
            c_0=succ,
        )

        out = {
            'l': l,
            'two_body': two_body,
            'Q': Q,
            'k': k,
            'succ': succ,
            'succ_kpq': succ_kpq,
            'ReIm': re_im,
            'p': p,
            'q': q,
            'AB': ab,
            'term': term,
            'anc': anc,
            'alpha': alpha,
            'psia': psia,
            'psib': psib,
        }
        return out
