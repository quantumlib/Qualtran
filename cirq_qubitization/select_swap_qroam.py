from typing import Tuple, Sequence, Optional, List
from functools import cached_property
import numpy as np
import cirq
from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.qrom import QROM
from cirq_qubitization.swap_network import SwapWithZeroGate


class SelectSwapQROM(GateWithRegisters):
    """Gate to load data[l] in the target register when the selection register stores integer l.

    Let
        N:= Number of data elements to load a
        b:= Bit-length of the target register in which data elements should be loaded.

    The `SelectSwapQROM` is a hybrid of the following two existing primitives:

        * Unary Iteration based `cirq_qubitization.QROM` requires O(N) T-gates to load `N` data
        elements into a b-bit target register. Note that the T-complexity is independent of `b`.
        * `cirq_qubitization.SwapWithZeroGate` can swap a `b` bit register indexed `x` with a `b`
        bit register at index `0` using O(b) T-gates, if the selection register stores integer `x`.
        Note that the swap complexity is independent of the iteration length `N`.

    The `SelectSwapQROM` uses square root decomposition by combining the above two approaches to
    further optimize the T-gate complexity of loading `N` data elements, each into a `b` bit
    target register as follows:

        * Divide the `N` data elements into batches of size `B` (a variable) and
        load each batch simultaneously into `B` distinct target registers using the conventional
        QROM. This has T-complexity `O(N / B)`.
        * Use `SwapWithZeroGate` to swap the `i % B`'th target register in batch number `i / B`
        to load `data[i]` in the 0'th target register. This has T-complexity `O(B * b)`.

    This, the final T-complexity of `SelectSwapQROM` is `O(B * b + N / B)` T-gates; where `B` is
    the block-size with an optimal value of `O(sqrt(N / b))`.

    This improvement in T-complexity is achieved at the cost of using an additional `O(B * b)`
    ancilla qubits, with a nice property that these additional ancillas can be `dirty`; i.e.
    they don't need to start in the |0> state and thus can be borrowed from other parts of the
    algorithm. The state of these dirty ancillas would be unaffected after the operation has
    finished.

    For more details, see the reference below:

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
            Low, Kliuchnikov, Schaeffer. 2018.
    """

    def __init__(
        self,
        *data: Sequence[int],
        target_bitsizes: Optional[Sequence[int]] = None,
        block_size: Optional[int] = None,
    ):
        # Validate input.
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        if target_bitsizes is None:
            target_bitsizes = [max(d).bit_length() for d in data]
        assert len(target_bitsizes) == len(data)
        assert all(t >= max(d).bit_length() for t, d in zip(target_bitsizes, data))
        self._iteration_length = len(data[0])
        if block_size is None:
            block_size = int(np.ceil(np.sqrt(self._iteration_length / sum(target_bitsizes))))
        assert 0 < block_size <= self._iteration_length
        # Figure out optimal value of block_size and register sizes accordingly.
        self._block_size = block_size
        self._num_blocks = int(np.ceil(self._iteration_length / self.block_size))
        batched_data = []
        batched_target_bitsizes = []
        for d, bitsize in zip(data, target_bitsizes):
            # Each original data array is split into `r` different sequences, each of length `q`.
            current_batch = []
            for st in range(self.block_size):
                curr_data = list(d[st :: self.block_size])
                curr_data += [0] * (self.num_blocks - len(curr_data))
                assert len(curr_data) == self.num_blocks
                current_batch.append(curr_data)
            batched_data.append(current_batch)
            batched_target_bitsizes.append([bitsize] * self.block_size)
        self.qrom = QROM(
            *[d for batch in batched_data for d in batch],
            target_bitsizes=[b for batch in batched_target_bitsizes for b in batch],
        )
        self.swap_with_zero = SwapWithZeroGate(
            (self.block_size - 1).bit_length(), sum(target_bitsizes), self.block_size
        )
        self._iteration_length = len(data[0])
        self._batched_data = tuple(tuple(tuple(d) for d in batch) for batch in batched_data)
        self._batched_target_bitsizes = batched_target_bitsizes
        self._selection_bitsizes = tuple(
            (L - 1).bit_length() for L in [self.num_blocks, self.block_size]
        )

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(
            selection_q=self._selection_bitsizes[0], selection_r=self._selection_bitsizes[1]
        )

    @cached_property
    def selection_ancillas(self) -> Registers:
        return Registers.build(selection_ancilla=self._selection_bitsizes[0])

    @cached_property
    def target_registers(self) -> Registers:
        clean_output = {}
        for i, target_bitsize in enumerate(self._batched_target_bitsizes):
            clean_output[f'target{i}'] = target_bitsize[0]
        return Registers.build(**clean_output)

    @cached_property
    def target_dirty_ancillas(self) -> Registers:
        dirty_ancillas = {}
        for i, target_bitsize in enumerate(self._batched_target_bitsizes):
            for b, bitsize in enumerate(target_bitsize):
                dirty_ancillas[f'target_dirty_ancilla_{i}_{b}'] = bitsize
        return Registers.build(**dirty_ancillas)

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [
                *self.selection_registers,
                *self.selection_ancillas,
                *self.target_registers,
                *self.target_dirty_ancillas,
            ]
        )

    @cached_property
    def iteration_length(self) -> int:
        return self._iteration_length

    @property
    def batched_data(self) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
        return self._batched_data

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def __repr__(self) -> str:
        data_repr = ",".join(repr(sum(batch, ())) for batch in self.batched_data)
        target_repr = repr([batch[0] for batch in self._batched_target_bitsizes])
        return f"cirq_qubitization.SelectSwapQROM({data_repr}, target_bitsizes={target_repr})"

    def decompose_from_registers(
        self,
        selection_q: Sequence[cirq.Qid],
        selection_r: Sequence[cirq.Qid],
        selection_ancilla: Sequence[cirq.Qid],
        **targets: Sequence[cirq.Qid],
    ) -> cirq.OP_TREE:
        # Parse and construct appropriate qubit registers.
        qrom_dirty_targets: List[cirq.Qid] = []
        swap_dirty_targets = [[] for _ in range(self.block_size)]
        for data_idx in range(len(self.batched_data)):
            for batch_idx in range(self.block_size):
                qubits = list(targets[f'target_dirty_ancilla_{data_idx}_{batch_idx}'])
                qrom_dirty_targets += qubits
                swap_dirty_targets[batch_idx] += qubits
        for batch_idx in range(self.block_size):
            assert len(swap_dirty_targets[batch_idx]) == sum(
                batch[0] for batch in self._batched_target_bitsizes
            )
        clean_targets = self.target_registers.merge_qubits(**targets)
        # Construct the unique operations.
        qrom_op = self.qrom.on_registers(
            selection=selection_q,
            ancilla=selection_ancilla,
            **self.qrom.target_registers.split_qubits(qrom_dirty_targets),
        )
        swap_with_zero_op = self.swap_with_zero.on_registers(
            selection=selection_r,
            **{
                self.swap_with_zero.target_registers[i].name: t
                for i, t in enumerate(swap_dirty_targets)
            },
        )
        cnot_op = cirq.Moment(cirq.CNOT(s, t) for s, t in zip(swap_dirty_targets[0], clean_targets))

        # Yield the operations in correct order.
        yield qrom_op
        yield swap_with_zero_op
        yield cnot_op
        yield swap_with_zero_op**-1
        yield qrom_op**-1
        yield swap_with_zero_op
        yield cnot_op
        yield swap_with_zero_op**-1

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        wire_symbols = ["In_q"] * self._selection_bitsizes[0]
        wire_symbols += ["In_r"] * self._selection_bitsizes[1]
        wire_symbols += ["Anc"] * self.selection_ancillas.bitsize
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROAM_{i}"] * target.bitsize
        for i, target_bitsize in enumerate(self._batched_target_bitsizes):
            for b, bitsize in enumerate(target_bitsize):
                wire_symbols += [f"TAnc_{i}_{b}"] * bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __eq__(self, other: 'SelectSwapQROM') -> bool:
        return (
            self.batched_data == other.batched_data
            and self._batched_target_bitsizes == other._batched_target_bitsizes
        )
