"""Gates for preparing coefficient states.

In section III.D. of the [Linear T paper](https://arxiv.org/abs/1805.03662) the authors introduce
a technique for initializing a state with $L$ unique coefficients (provided by a classical
database) with a number of T gates scaling as 4L + O(log(1/eps)) where eps is the
largest absolute error that one can tolerate in the prepared amplitudes.
"""


from typing import List, Sequence

import cirq
from openfermion.circuits.lcu_util import preprocess_lcu_coefficients_for_reversible_sampling

from cirq_qubitization.arithmetic_gates import LessThanEqualGate
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq_qubitization.qrom import QROM
from cirq_qubitization.swap_network import MultiTargetCSwap


class GenericSubPrepare(GateWithRegisters):
    r"""Initialize a state with $L$ unique coefficients.

    In particular, we take the zero state to:

    $$
    \sum_{\ell=0}^{L-1} \sqrt{p_\ell} |\ell\rangle |\mathrm{temp}_\ell\rangle
    $$

    where the probabilities $p_\ell$ are $\mu$-bit binary approximations to the true values and
    where the temporary register must be treated with care, see the details in Section III.D. of
    the reference.

    The preparation is equivalent to [classical alias sampling](https://en.wikipedia.org/wiki/Alias_method):
    we sample `l` with probability `p[l]` by first selecting `l` uniformly at random and then
    returning it with probability `keep[l] / 2**mu`; otherwise returning `alt[l]`.

    Registers:
        selection: The input/output register $|\ell\rangle$ of size lg(L) where the desired
            coefficient state is prepared.
        temp: Work space comprised of sub registers:
            1.) `sigma`: A mu-sized register containing uniform probabilities for comparison
                against `keep`.
            2.) `alt`: A lg(L)-sized register of alternate indices.
            3.) `keep`: a mu-sized register of probabilities of keeping the initially sampled index.
            4.) one bit for the result of the comparison.
        selection_ancilla: Ancilla space for QROM loading.

    This gate corresponds to the following operations:
     - UNIFORM_L on the selection register
     - H^mu on the sigma register
     - QROM addressed by the selection register into the alt and keep registers.
     - LessThanEqualGate comparing the keep and sigma registers.
     - Coherent swap between the selection register and alt register if the comparison
       returns True.

    Total space will be (2 * log(L) + 2 mu + 1) work qubits + log(L) ancillas for QROM.
    The 1 ancilla in work qubits is for the `LessThanEqualGate` followed by coherent swap.

    Args:
        lcu_probabilities: The LCU coefficients.
        probability_epsilon: The desired accuracy to represent each probability
            (which sets mu size and keep/alt integers).
            See `openfermion.circuits.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
            for more information.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 11.
    """

    def __init__(
        self, lcu_probabilities: List[float], *, probability_epsilon: float = 1.0e-5
    ) -> None:
        self._lcu_probs = lcu_probabilities
        self._probability_epsilon = probability_epsilon
        self._selection_bitsize = (len(self._lcu_probs) - 1).bit_length()
        (self._alt, self._keep, self._mu) = preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=lcu_probabilities, epsilon=probability_epsilon
        )

    @property
    def selection_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def sigma_mu_bitsize(self) -> int:
        return self._mu

    @property
    def alternates_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def keep_bitsize(self) -> int:
        return self._mu

    @property
    def temp_bitsize(self) -> int:
        return self.sigma_mu_bitsize + self.alternates_bitsize + self.keep_bitsize + 1

    @property
    def ancilla_bitsize(self) -> int:
        return self._selection_bitsize

    @property
    def registers(self) -> Registers:
        return Registers.build(
            selection=self.selection_bitsize,
            temp=self.temp_bitsize,
            selection_ancilla=self.ancilla_bitsize,
        )

    def __repr__(self) -> str:
        return (
            f"cirq_qubitization.GenericSubPrepare("
            f"lcu_probabilities={self._lcu_probs},"
            f"probability_epsilon={self._probability_epsilon}"
            f")"
        )

    def decompose_from_registers(
        self,
        selection: Sequence[cirq.Qid],
        temp: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:

        sigma_mu, alt, keep, less_than_equal = (
            temp[: self._mu],
            temp[self._mu : self._mu + self._selection_bitsize],
            temp[-(self._mu + 1) : -1],
            temp[-1],
        )

        yield PrepareUniformSuperposition(len(self._lcu_probs)).on(*selection, selection_ancilla[0])
        qrom = QROM(self._alt, self._keep, target_bitsizes=[len(alt), len(keep)])
        yield qrom.on_registers(
            selection=selection, ancilla=selection_ancilla[1:], target0=alt, target1=keep
        )
        yield cirq.H.on_each(*sigma_mu)
        yield LessThanEqualGate([2] * self._mu, [2] * self._mu).on(
            *sigma_mu, *keep, less_than_equal
        )
        yield MultiTargetCSwap(self.selection_bitsize).on_registers(
            control=less_than_equal, target_x=alt, target_y=selection
        )
