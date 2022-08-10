import abc
import dataclasses
import sys
from typing import Sequence, Dict, Iterable, List, Union

import cirq

assert sys.version_info > (3, 6), "https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-pep468"


@dataclasses.dataclass(frozen=True)
class Register:
    name: str
    bitsize: int


@cirq.value_equality()
class Registers:
    def __init__(self, registers: Iterable[Register]):
        self._registers = tuple(registers)
        self._register_dict = {r.name: r for r in self._registers}
        if len(self._registers) != len(self._register_dict):
            raise ValueError("Please provide unique register names.")

    def _value_equality_values_(self):
        return self._registers

    @classmethod
    def build(cls, **registers: int):
        return cls(Register(name=k, bitsize=v) for k, v in registers.items())

    def at(self, i: int):
        return self._registers[i]

    def __getitem__(self, name: str):
        return self._register_dict[name]

    def __iter__(self):
        yield from self._registers

    def split_qubits(self, qubits: Sequence[cirq.Qid]) -> Dict[str, Sequence[cirq.Qid]]:
        qubit_regs = {}
        base = 0
        for reg in self:
            qubit_regs[reg.name] = qubits[base : base + reg.bitsize]
            base += reg.bitsize
        return qubit_regs

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


class GateWithRegisters(cirq.Gate, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def registers(self) -> Registers:
        ...

    def _num_qubits_(self) -> int:
        return sum(reg.bitsize for reg in self.registers)

    @abc.abstractmethod
    def decompose_from_registers(self, **qubit_regs: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        ...

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        qubit_regs = self.registers.split_qubits(qubits)
        yield from self.decompose_from_registers(**qubit_regs)

    def on_registers(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid]]) -> cirq.GateOperation:
        return self.on(*self.registers.merge_qubits(**qubit_regs))
