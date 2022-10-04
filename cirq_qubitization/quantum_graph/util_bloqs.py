import abc
import re
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Sequence, Union, List, overload, Any, Tuple, Set

import cirq
import pydot

from cirq_qubitization.gate_with_registers import Registers, Register
from cirq_qubitization.quantum_graph.fancy_registers import (
    SplitRegister,
    JoinRegister,
    ApplyFRegister,
)
from cirq_qubitization.quantum_graph.bloq import Bloq


@dataclass(frozen=True)
class Split(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([SplitRegister(name='sss', bitsize=self.bitsize)])


@dataclass(frozen=True)
class Join(Bloq):
    bitsize: int

    @cached_property
    def registers(self) -> Registers:
        return Registers([JoinRegister(name='jjj', bitsize=self.bitsize)])
