"""Gates expected to be found in `cirq_algos.swap_network` that are wrappers around bloqs."""

from typing import Sequence

import cirq

from cirq_qubitization.bloq_algos.basic_gates import CSwap
from cirq_qubitization.bloq_algos.swap_network import CSwapApprox, SwapWithZero
from cirq_qubitization.quantum_graph.cirq_conversion import BloqAsCirqGate
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister

# for reprs
_MY_NAMESPACE = 'cirq_qubitization.bloq_algos.swap_network_cirq'


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
        return f"{_MY_NAMESPACE}.MultiTargetCSwap({self._target_bitsize})"


MultiTargetCSwap.__doc__ = CSwap.__doc__


class MultiTargetCSwapApprox(BloqAsCirqGate):
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
        return f"{_MY_NAMESPACE}.MultiTargetCSwapApprox({self._target_bitsize})"


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
