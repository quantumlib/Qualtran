r"""Gates for Qubitizing the Hubbard Model.

This follows section V. of the [Linear T Paper](https://arxiv.org/abs/1805.03662).

The Hubbard model is a special case of the electronic structure Hamiltonian
restricted to spins on a planar grid.

$$
H = -t \sum_{\langle p,q \rangle, \sigma} a_{p,\sigma}^\dagger a_{q,\sigma}
    + \frac{u}{2} \sum_{p,\alpha\ne\beta} n_{p, \alpha} n_{p, \beta}
$$

Under the Jordan-Wigner transformation this is

$$
\def\Zvec{\overrightarrow{Z}}
\def\hop#1{#1_{p,\sigma} \Zvec #1_{q,\sigma}}
H = -\frac{t}{2} \sum_{\langle p,q \rangle, \sigma} (\hop{X} + \hop{Y})
  + \frac{u}{8} \sum_{p,\alpha\ne\beta} Z_{p,\alpha}Z_{p,\beta}
  - \frac{u}{4} \sum_{p,\sigma} Z_{p,\sigma} + \frac{uN}{4}\mathbb{1}
$$


This model consists of a PREPARE and SELECT operation where our selection operation has indices
for p, alpha, q, and beta as well as two indicator bits U and V. There are four cases considered
in both the PREPARE and SELECT operations corresponding to the terms in the Hamiltonian:

 - $U=1$, single-body Z
 - $V=1$, spin-spin ZZ term
 - $p<q$, XZX term
 - $p>q$, YZY term.

See the documentation for `PrepareHubbard` and `SelectHubbard` for details.
"""
from functools import cached_property
from typing import Sequence

import cirq
from attrs import frozen

from cirq_qubitization import ApplyGateToLthQubit, MultiTargetCSwap, SelectedMajoranaFermionGate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers


@frozen
class SelectHubbard(GateWithRegisters):
    r"""The SELECT operation optimized for the 2D Hubbard model.

    In contrast to the arbitrary chemistry Hamiltonian, we:
     - explicitly consider the two dimensions of indices to permit optimization of the circuits.
     - dispense with the `theta` parameter.

    If neither `U` nor `V` is set we apply the kinetic terms of the Hamiltonian:

    $$
    -\hop{X} \quad p < q \\
    -\hop{Y} \quad p > q
    $$

    If U is set we know $(p,\alpha)=(q,\beta)$ and apply the single-body term: $-Z_{p,\alpha}$.
    If V is set we know $p=q, \alpha=0$, and $\beta=1$ and apply the spin term:
    $Z_{p,\alpha}Z_{p,\beta}$


    Args:
        x_dim: the number of sites in the x axis.
        y_dim: the number of site along the y axis.

    Registers:
        control[1]: A control bit for the entire gate.
        U[1]: Whether we're applying the single-site part of the potential.
        V[1]: Whether we're applying the pairwise part of the potential.
        p_x: First set of site indices, x component.
        p_y: First set of site indices, y component.
        alpha[1]: First set of sites' spin indicator.
        q_x: Second set of site indices, x component.
        q_y: Second set of site indices, y component.
        beta[1]: Second set of sites' spin indicator.
        target: The system register to apply the select operation.
        ancilla: Work space.

    References:
        Section V. and Fig. 19 of https://arxiv.org/abs/1805.03662.
    """

    x_dim: int
    y_dim: int

    @cached_property
    def registers(self) -> Registers:
        x_dim = self.x_dim
        y_dim = self.y_dim
        sys_size = x_dim * y_dim * 2
        return Registers.build(
            control=1,
            U=1,
            V=1,
            p_x=(x_dim - 1).bit_length(),
            p_y=(y_dim - 1).bit_length(),
            alpha=1,
            q_x=(x_dim - 1).bit_length(),
            q_y=(y_dim - 1).bit_length(),
            beta=1,
            target=sys_size,
            ancilla=(x_dim - 1).bit_length() + (y_dim - 1).bit_length() + 2,
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
        (control, U, V, alpha, beta) = control + U + V + alpha + beta

        # TODO: this is broken without multi-select
        p_inds = p_x + p_y + (alpha,)
        q_inds = q_x + q_y + (beta,)
        yield SelectedMajoranaFermionGate.make_on(
            target_gate=cirq.Y,
            control=control,
            selection=p_inds,
            ancilla=ancilla[:-1],
            accumulator=ancilla[-1],
            target=target,
        )
        yield MultiTargetCSwap.make_on(control=V, target_x=p_inds, target_y=q_inds)
        yield SelectedMajoranaFermionGate.make_on(
            target_gate=cirq.X,
            control=control,
            selection=q_inds,
            ancilla=ancilla[:-1],
            accumulator=ancilla[-1],
            target=target,
        )
        yield MultiTargetCSwap.make_on(control=V, target_x=p_inds, target_y=q_inds)
        yield cirq.S(control) ** -1  # Fix errant i from XY=iZ
        yield cirq.CZ(control, U)  # Fix errant -1 from multiple pauli applications

        yield ApplyGateToLthQubit.make_on(
            nth_gate=lambda n: cirq.Z if n & 1 else cirq.I,
            control=[control, V],
            selection=q_inds,
            ancilla=ancilla[:],
            target=target,
        )
