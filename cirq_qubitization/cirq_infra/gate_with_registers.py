import abc
import sys
from typing import Dict, Iterable, List, overload, Sequence, Tuple, Union

import attrs
import cirq
import numpy as np

assert sys.version_info > (3, 6), "https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468"


@attrs.frozen
class Register:
    name: str
    bitsize: int


class Registers:
    def __init__(self, registers: Iterable[Register]):
        self._registers = tuple(registers)
        self._register_dict = {r.name: r for r in self._registers}
        if len(self._registers) != len(self._register_dict):
            raise ValueError("Please provide unique register names.")

    def __repr__(self):
        return f'Registers({repr(self._registers)})'

    @property
    def bitsize(self) -> int:
        return sum(reg.bitsize for reg in self)

    @classmethod
    def build(cls, **registers: int) -> 'Registers':
        return cls(Register(name=k, bitsize=v) for k, v in registers.items())

    @overload
    def __getitem__(self, key: int) -> Register:
        pass

    @overload
    def __getitem__(self, key: str) -> Register:
        pass

    @overload
    def __getitem__(self, key: slice) -> 'Registers':
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Registers(self._registers[key])
        elif isinstance(key, int):
            return self._registers[key]
        elif isinstance(key, str):
            return self._register_dict[key]
        else:
            raise IndexError(f"key {key} must be of the type str/int/slice.")

    def __contains__(self, item: str) -> bool:
        return item in self._register_dict

    def __iter__(self):
        yield from self._registers

    def __len__(self) -> int:
        return len(self._registers)

    def split_qubits(self, qubits: Sequence[cirq.Qid]) -> Dict[str, Sequence[cirq.Qid]]:
        qubit_regs = {}
        base = 0
        for reg in self:
            qubit_regs[reg.name] = qubits[base : base + reg.bitsize]
            base += reg.bitsize
        return qubit_regs

    def split_integer(self, n: int) -> Dict[str, int]:
        """Extracts integer values of individual qubit registers, given a bit-packed integer.

        Considers the given integer as a binary bit-packed representation of `self.bitsize` bits,
        with `n_bin[0 : self[0].bitsize]` representing the integer for first register,
        `n_bin[self[0].bitsize : self[1].bitsize]` representing the integer for second
        register and so on. Here `n_bin` is the `self.bitsize` length binary representation of
        input integer `n`.
        """
        qubit_vals = {}
        base = 0
        n_bin = f"{n:0{self.bitsize}b}"
        for reg in self:
            qubit_vals[reg.name] = int(n_bin[base : base + reg.bitsize], 2)
            base += reg.bitsize
        return qubit_vals

    def merge_qubits(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> List[cirq.Qid]:
        ret: List[cirq.Qid] = []
        for reg in self:
            assert reg.name in qubit_regs, "All qubit registers must pe present"
            qubits = qubit_regs[reg.name]
            qubits = [qubits] if isinstance(qubits, cirq.Qid) else qubits
            assert len(qubits) == reg.bitsize, f"{reg.name} register must of length {reg.bitsize}."
            ret += qubits
        return ret

    def get_named_qubits(self) -> Dict[str, List[cirq.Qid]]:
        def qubits_for_reg(name: str, bitsize: int):
            return (
                [cirq.NamedQubit(f"{name}")]
                if bitsize == 1
                else cirq.NamedQubit.range(bitsize, prefix=name)
            )

        return {reg.name: qubits_for_reg(reg.name, reg.bitsize) for reg in self}

    def __eq__(self, other) -> bool:
        return self._registers == other._registers

    def __hash__(self):
        return hash(self._registers)


@attrs.frozen
class SelectionRegister(Register):
    iteration_length: int = attrs.field()

    @iteration_length.validator
    def validate_iteration_length(self, attribute, value):
        if not (0 <= value <= 2**self.bitsize):
            raise ValueError(f'iteration length must be in range [0, 2^{self.bitsize}]')


class SelectionRegisters(Registers):
    """Registers used to represent SELECT registers for various LCU methods.

    LCU methods often make use of coherent for-loops via UnaryIteration, iterating over a range
    of values stored as a superposition over the `SELECT` register. The `SelectionRegisters` class
    is used to represent such SELECT registers. In particular, it provides two additional features
    on top of the regular `Registers` class:

    - For each selection register, we store the iteration length corresponding to that register
        along with its size.
    - We provide a default way of "flattening out" a composite index represented by a tuple of
        values stored in multiple input selection registers to a single integer that can be used
        to index a flat target register.
    """

    def __init__(self, registers: Iterable[SelectionRegister]):
        super().__init__(registers)
        self.iteration_lengths = tuple([reg.iteration_length for reg in registers])
        self._suffix_prod = np.multiply.accumulate(self.iteration_lengths[::-1])[::-1]
        self._suffix_prod = np.append(self._suffix_prod, [1])

    def to_flat_idx(self, *selection_vals: int) -> int:
        """Flattens a composite index represented by a Tuple[int, ...] to a single output integer.

        For example:

        1) We can flatten a 2D for-loop as follows
        >>> for x in range(N):
        >>>     for y in range(M):
        >>>         flat_idx = x * M + y

        2) Similarly, we can flatten a 3D for-loop as follows
        >>> for x in range(N):
        >>>     for y in range(M):
        >>>         for z in range(L):
        >>>             flat_idx = x * M * L + y * L + z

        This is a general version of the mapping function described in Eq.45 of
        https://arxiv.org/abs/1805.03662
        """
        assert len(selection_vals) == len(self)
        return sum(v * self._suffix_prod[i + 1] for i, v in enumerate(selection_vals))

    @property
    def total_iteration_size(self) -> int:
        return np.product(self.iteration_lengths)

    @classmethod
    def build(cls, **registers: Tuple[int, int]) -> 'SelectionRegisters':
        return cls(
            [
                SelectionRegister(name=k, bitsize=v[0], iteration_length=v[1])
                for k, v in registers.items()
            ]
        )

    @overload
    def __getitem__(self, key: int) -> SelectionRegister:
        pass

    @overload
    def __getitem__(self, key: str) -> SelectionRegister:
        pass

    @overload
    def __getitem__(self, key: slice) -> SelectionRegister:
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return SelectionRegisters(self._registers[key])
        elif isinstance(key, int):
            return self._registers[key]
        elif isinstance(key, str):
            return self._register_dict[key]
        else:
            raise IndexError(f"key {key} must be of the type str/int/slice.")


class GateWithRegisters(cirq.Gate, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def _num_qubits_(self) -> int:
        return self.registers.bitsize

    @abc.abstractmethod
    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        ...

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.decompose_from_registers(**qubit_regs)

    def on_registers(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> cirq.Operation:
        return self.on(*self.registers.merge_qubits(**qubit_regs))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Default diagram info that uses register names to name the boxes in multi-qubit gates.

        Descendants can override this method with more meaningful circuit diagram information.
        """
        wire_symbols = []
        for reg in self.registers:
            wire_symbols += [reg.name] * reg.bitsize

        wire_symbols[0] = self.__class__.__name__
        return cirq.CircuitDiagramInfo(wire_symbols=wire_symbols)
