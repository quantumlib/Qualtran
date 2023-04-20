from typing import Sequence, TYPE_CHECKING

import cirq

from cirq_qubitization.bloq_algos.basic_gates import CSwap
from cirq_qubitization.bloq_algos.swap_network import CSwapApprox, SwapWithZero
from cirq_qubitization.quantum_graph.cirq_conversion import BloqAsCirqGate

if TYPE_CHECKING:
    from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister


class MultiTargetCSwap(BloqAsCirqGate):
    def __init__(self, target_bitsize: int):
        super().__init__(bloq=CSwap(bitsize=target_bitsize))

    @classmethod
    def make_on(cls, **quregs: Sequence[cirq.Qid]) -> cirq.Operation:
        """Helper constructor to automatically deduce bitsize attributes."""
        return cls(target_bitsize=len(quregs['x'])).on_registers(**quregs)

    @property
    def _target_bitsize(self) -> int:
        return self._bloq.bitsize

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@",) + ("swap_x",) * self._target_bitsize + ("swap_y",) * self._target_bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@",) + ("×(x)",) * self._target_bitsize + ("×(y)",) * self._target_bitsize
        )

    def __repr__(self) -> str:
        return f"cirq_qubitization.MultiTargetCSwap({self._target_bitsize})"


MultiTargetCSwap.__doc__ = CSwap.__doc__


class MultiTargetCSwapApprox(BloqAsCirqGate):
    """Approximately implements a multi-target controlled swap unitary using only 4 * n T-gates.

    Implements the unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$ such that the output state is
    correct up to a global phase factor of +1 / -1.

    This is useful when the incorrect phase can be absorbed in a garbage state of an algorithm; and
    thus ignored, see the reference for more details.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis](https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """

    def __init__(self, target_bitsize: int):
        super().__init__(bloq=CSwapApprox(bitsize=target_bitsize))

    @property
    def _target_bitsize(self) -> int:
        return self._bloq.bitsize

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(
                ("@(approx)",)
                + ("swap_x",) * self._target_bitsize
                + ("swap_y",) * self._target_bitsize
            )
        return cirq.CircuitDiagramInfo(
            ("@(approx)",) + ("×(x)",) * self._target_bitsize + ("×(y)",) * self._target_bitsize
        )

    def __repr__(self) -> str:
        return f"cirq_qubitization.MultiTargetCSwapApprox({self._target_bitsize})"


MultiTargetCSwapApprox.__doc__ = CSwapApprox.__doc__


def swz_reg_to_wires(reg: 'FancyRegister'):
    if reg.name == 'selection':
        return ["@(r⇋0)"] * reg.bitsize

    if reg.name == 'targets':
        symbs = []
        (n_target,) = reg.wireshape
        for i in range(n_target):
            symbs += [f"swap_{i}"] * reg.bitsize
        return symbs

    raise ValueError(f"Unknown register {reg}")


class SwapWithZeroGate(BloqAsCirqGate):
    def __init__(self, selection_bitsize: int, target_bitsize: int, n_target_registers: int):
        super().__init__(
            bloq=SwapWithZero(
                selection_bitsize=selection_bitsize,
                target_bitsize=target_bitsize,
                n_target_registers=n_target_registers,
            ),
            reg_to_wires=swz_reg_to_wires,
        )


SwapWithZeroGate.__doc__ = SwapWithZero.__doc__
