import numpy as np

from cirq_qubitization import t_complexity_protocol
from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegisters


class PrepareUniformSuperposition(Bloq):
    r"""Prepare a uniform superposition over an index l:

    $$
        |\psi\rangle = \frac{1}{\sqrt{n}}\sum_l^{n}|l\rangle,
    $$
    where $N = 2^k L$ and $L$ is an odd number. Can optionally be controlled on
    a single qubit register.

    Args:
        n: The number of terms to create the superposition over.
        num_controls: The number of controls, either 0 or 1.
        eps: Precision for Z-rotations in preparation. 1.0E-8 is about 30
            T-gates per rotation.

    Registers:
     - control: Optional control register.
     - target: target register of size k + log L.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 12.
    """

    def __init__(self, n: int, num_controls: int = 0, eps: 1e-8):
        target_bitsize = (n - 1).bit_length()
        self._K = 0
        while n > 1 and n % 2 == 0:
            self._K += 1
            n = n // 2
        self._L = int(n)
        self._logL = target_bitsize - self._K
        self._num_controls = num_controls
        self._eps = eps

    @property
    def registers(self):
        return FancyRegisters.build(control=self._num_controls, target=self._K + self._logL)

    def pretty_name(self) -> str:
        return "Uniform_{2^k L}"

    def t_complexity(self):
        rotation_cost = int(np.ceil(1.149 * np.log2(1.0 / self._eps) + 9.2))
        if self._num_controls == 0:
            num_t = 8 * self._logL + rotation_cost if self._L > 1 else 0
        else:
            num_t = 2 * self._K + 10 * self._logL + rotation_cost if self._L > 1 else 2 * self._K
        return t_complexity_protocol.TComplexity(t=num_t)