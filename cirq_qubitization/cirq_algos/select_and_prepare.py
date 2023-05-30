import abc
from functools import cached_property

from cirq_qubitization.cirq_infra.gate_with_registers import (
    GateWithRegisters,
    Registers,
    SelectionRegisters,
)


class SelectOracle(GateWithRegisters):
    @property
    @abc.abstractmethod
    def control_registers(self) -> Registers:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> SelectionRegisters:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Registers:
        ...

    @cached_property
    def registers(self) -> Registers:
        return Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )


class PrepareOracle(GateWithRegisters):
    @property
    @abc.abstractmethod
    def selection_registers(self) -> SelectionRegisters:
        ...

    @cached_property
    def junk_registers(self) -> Registers:
        return Registers([])

    @cached_property
    def registers(self) -> Registers:
        return Registers([*self.selection_registers, *self.junk_registers])
