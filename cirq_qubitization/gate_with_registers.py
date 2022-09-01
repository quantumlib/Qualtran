import abc
import dataclasses
import sys
from typing import Sequence, Dict, Iterable, List, Union, overload, Tuple

import cirq
import numpy as np

assert sys.version_info > (3, 6), "https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468"


@dataclasses.dataclass(frozen=True)
class Register:
    name: str
    bitsize: int


class Registers:
    def __init__(self, registers: Iterable[Register]):
        self._registers = tuple(registers)
        self._register_dict = {r.name: r for r in self._registers}
        if len(self._registers) != len(self._register_dict):
            raise ValueError("Please provide unique register names.")

    @property
    def bitsize(self) -> int:
        return sum(reg.bitsize for reg in self)

    def __repr__(self):
        return repr(self._registers)

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

    def __iter__(self) -> Iterable[Register]:
        yield from self._registers

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

    def merge_qubits(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> Sequence[cirq.Qid]:
        ret: List[cirq.Qid] = []
        for reg in self:
            assert reg.name in qubit_regs, "All qubit registers must pe present"
            qubits = qubit_regs[reg.name]
            qubits = [qubits] if isinstance(qubits, cirq.Qid) else qubits
            assert len(qubits) == reg.bitsize, f"{reg.name} register must of length {reg.bitsize}."
            ret += qubits
        return ret

    def get_named_qubits(self) -> Dict[str, Sequence[cirq.Qid]]:
        def qubits_for_reg(name: str, bitsize: int):
            return (
                [cirq.NamedQubit(f"{name}")]
                if bitsize == 1
                else cirq.NamedQubit.range(bitsize, prefix=name)
            )

        return {reg.name: qubits_for_reg(reg.name, reg.bitsize) for reg in self}

    def __eq__(self, other) -> bool:
        return self._registers == other._registers


def munge_classical_arrays(
    r: Registers, vals: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], int]:
    """Coerce classical arrays into the correct shape.

    This function will ensure `vals` is in a form that one would get from
    `cq_testing.get_classical_inputs`.

    This casts all items into 2d arrays, validates that the lengths are self-consistent,
    and validates that array widths are consistent with register bitwidths.

    Args:
        r: The registers used to validate the shape of arrays
        vals: The dictionary of ndarrays to parse.

    Returns:
        vals: The munged ndarrays
        n_states: The length of each array.
    """
    n_states = None  # to be filled in
    for k in vals.keys():
        arr = np.asarray(vals[k])

        if n_states is None:
            n_states = len(vals[k])

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]

        if arr.shape != (n_states, r[k].bitsize):
            raise ValueError(
                f"Incorrect shape for {k}: {arr.shape} (expected ({n_states}, {r[k].bitsize})"
            )

        vals[k] = arr

    if n_states is None:
        raise ValueError("No values were provided.")
    return vals, n_states


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

    def on_registers(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> cirq.GateOperation:
        return self.on(*self.registers.merge_qubits(**qubit_regs))

    def _apply_classical_from_registers(self, **vals: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def apply_classical(self, vals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        vals, _ = munge_classical_arrays(self.registers, vals)
        return self._apply_classical_from_registers(**vals)
