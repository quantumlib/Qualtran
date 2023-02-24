from functools import cached_property
from typing import List, Optional, Sequence, Tuple

import cirq
import numpy as np

from cirq_qubitization.gate_with_registers import GateWithRegisters, Registers
from cirq_qubitization.qrom import QROM
from cirq_qubitization.qubit_manager import qalloc_borrow, qalloc_clean
from cirq_qubitization.swap_network import SwapWithZeroGate


def find_optimal_log_block_size(iteration_length: int, target_bitsize: int) -> int:
    """Find optimal block size, which is a power of 2, for SelectSwapQROM.

    This functions returns the optimal `k` s.t.
        * k is in an integer and k >= 0.
        * iteration_length/2^k + target_bitsize*(2^k - 1) is minimized.
    The corresponding block size for SelectSwapQROM would be 2^k.
    """
    k = 0.5 * np.log2(iteration_length / target_bitsize)
    if k < 0:
        return 1

    def value(k: List[int]):
        return iteration_length / np.power(2, k) + target_bitsize * (np.power(2, k) - 1)

    k_int = [np.floor(k), np.ceil(k)]  # restrict optimal k to integers
    return int(k_int[np.argmin(value(k_int))])  # obtain optimal k


class SelectSwapQROM(GateWithRegisters):
    """Gate to load data[l] in the target register when the selection register stores integer l.

    Let
        N:= Number of data elements to load.
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
        """Initializes SelectSwapQROM

        For a single data sequence of length `N`, maximum target bitsize `b` and block size `B`;
        SelectSwapQROM requires:
            - Selection register & ancilla of size `logN` for QROM data load.
            - 1 clean target register of size `b`.
            - `B` dirty target registers, each of size `b`.

        Similarly, to load `M` such data sequences, `SelectSwapQROM` requires:
            - Selection register & ancilla of size `logN` for QROM data load.
            - 1 clean target register of size `sum(target_bitsizes)`.
            - `B` dirty target registers, each of size `sum(target_bitsizes)`.

        Args:
            data: Sequence of integers to load in the target register. If more than one sequence
                is provided, each sequence must be of the same length.
            target_bitsizes: Sequence of integers describing the size of target register for each
                data sequence to load. Defaults to `max(data[i]).bit_length()` for each i.
            block_size(B): Load batches of `B` data elements in each iteration of traditional QROM
                (N/B iterations required). Complexity of SelectSwap QROAM scales as
                `O(B * b + N / B)`, where `B` is the block_size. Defaults to optimal value of
                 `O(sqrt(N / b))`.
        """
        # Validate input.
        if len(set(len(d) for d in data)) != 1:
            raise ValueError("All data sequences to load must be of equal length.")
        if target_bitsizes is None:
            target_bitsizes = [max(d).bit_length() for d in data]
        assert len(target_bitsizes) == len(data)
        assert all(t >= max(d).bit_length() for t, d in zip(target_bitsizes, data))
        self._num_sequences = len(data)
        self._target_bitsizes = target_bitsizes
        self._iteration_length = len(data[0])
        if block_size is None:
            # Figure out optimal value of block_size
            block_size = 2 ** find_optimal_log_block_size(
                self._iteration_length, sum(target_bitsizes)
            )
        assert 0 < block_size <= self._iteration_length
        self._block_size = block_size
        self._num_blocks = int(np.ceil(self._iteration_length / self.block_size))
        self.selection_q, self.selection_r = tuple(
            (L - 1).bit_length() for L in [self.num_blocks, self.block_size]
        )
        self._data = tuple(tuple(d) for d in data)

    @cached_property
    def selection_registers(self) -> Registers:
        return Registers.build(selection=self.selection_q + self.selection_r)

    @cached_property
    def target_registers(self) -> Registers:
        clean_output = {}
        for sequence_id in range(self._num_sequences):
            clean_output[f'target{sequence_id}'] = self._target_bitsizes[sequence_id]
        return Registers.build(**clean_output)

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.target_registers])

    @cached_property
    def iteration_length(self) -> int:
        return self._iteration_length

    @property
    def data(self) -> Tuple[Tuple[int, ...], ...]:
        return self._data

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def __repr__(self) -> str:
        data_repr = ','.join(repr(d) for d in self.data)
        target_repr = repr(self._target_bitsizes)
        return (
            f"cirq_qubitization.SelectSwapQROM("
            f"{data_repr}, "
            f"target_bitsizes={target_repr}, "
            f"block_size={self.block_size})"
        )

    def decompose_from_registers(
        self, selection: Sequence[cirq.Qid], **targets: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        # Divide each data sequence and corresponding target registers into
        # `self.num_blocks` batches of size `self.block_size`.
        qrom_data: List[Tuple[int, ...]] = []
        qrom_target_bitsizes: List[int] = []
        ordered_target_qubits: List[cirq.Qid] = []
        for block_id in range(self.block_size):
            for sequence_id in range(self._num_sequences):
                data = self.data[sequence_id]
                target_bitsize = self._target_bitsizes[sequence_id]
                ordered_target_qubits.extend(qalloc_borrow(target_bitsize))
                data_for_current_block = data[block_id :: self.block_size]
                if len(data_for_current_block) < self.num_blocks:
                    zero_pad = (0,) * (self.num_blocks - len(data_for_current_block))
                    data_for_current_block = data_for_current_block + zero_pad
                qrom_data.append(data_for_current_block)
                qrom_target_bitsizes.append(target_bitsize)
        # Construct QROM, SwapWithZero and CX operations using the batched data and qubits.
        k = (self.block_size - 1).bit_length()
        q, r = selection[: self.selection_q], selection[self.selection_q :]
        qrom_gate = QROM(*qrom_data, target_bitsizes=qrom_target_bitsizes)
        qrom_op = qrom_gate.on_registers(
            selection=q, **qrom_gate.target_registers.split_qubits(ordered_target_qubits)
        )
        swap_with_zero_gate = SwapWithZeroGate(k, self.target_registers.bitsize, self.block_size)
        swap_with_zero_op = swap_with_zero_gate.on_registers(
            selection=r, **swap_with_zero_gate.target_registers.split_qubits(ordered_target_qubits)
        )
        clean_targets = self.target_registers.merge_qubits(**targets)
        cnot_op = cirq.Moment(cirq.CNOT(s, t) for s, t in zip(ordered_target_qubits, clean_targets))
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
        wire_symbols = ["In_q"] * self.selection_q
        wire_symbols += ["In_r"] * self.selection_r
        for i, target in enumerate(self.target_registers):
            wire_symbols += [f"QROAM_{i}"] * target.bitsize
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)

    def __eq__(self, other: 'SelectSwapQROM') -> bool:
        return (
            self.data == other.data
            and self._target_bitsizes == other._target_bitsizes
            and self.block_size == other.block_size
        )
