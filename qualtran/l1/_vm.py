#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import abc
from collections import Counter
from typing import Any, Dict, Iterable, List

import attrs
import numpy as np

import qualtran.testing as qlt_testing
from qualtran import Bloq, CompositeBloq, Register


def _mock_return_values(regs: Iterable[Register]):
    soqdict: Dict[str, Any] = {}

    # Initialize multi-dimensional dictionary values.
    for reg in regs:
        # val = next(iter(reg.dtype.get_classical_domain()))
        # reg.dtype.assert_valid_classical_val(val)

        if reg.shape:
            soqdict[reg.name] = np.empty(reg.shape, dtype=object)
        else:
            soqdict[reg.name] = object()

    return soqdict


@attrs.mutable
class StackFrame:
    bloq_str: str
    qlocals: Dict[str, Any] = attrs.field(factory=dict)


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_summary(self) -> str: ...


@attrs.frozen
class UnsupportedAtomicBloqProblem(Problem):
    bloq: 'Bloq'

    def get_summary(self) -> str:
        return f' {self.bloq!r} is not a supported atomic bloq.'


@attrs.mutable(kw_only=True)
class StandardQualtranArchitectureAgnosticVirtualMachine:
    frames: List[StackFrame] = attrs.field(factory=list)
    problems: List[Problem] = attrs.field(factory=list)
    n_calls: int = 0
    n_atoms: int = 0

    def qcall(self, cbloq: 'CompositeBloq', **soqs):
        self.frames.append(StackFrame(bloq_str=str(cbloq)))
        self.n_calls += 1
        qlt_testing.assert_valid_cbloq(cbloq)
        for binst, in_soqs, old_out_soqs in cbloq.iter_bloqsoqs():
            self.execute(binst.bloq, **in_soqs)
        self.frames.pop(-1)

    def qatom(self, bloq: 'Bloq', **soqs):
        self.n_atoms += 1

    def execute(self, bloq: 'Bloq', **soqs):
        if isinstance(bloq, CompositeBloq):
            return self.qcall(bloq, **soqs)

        if self.bloq_in_isa(bloq):
            return self.qatom(bloq, **soqs)

        if isinstance(bloq, Bloq):
            self.problems.append(UnsupportedAtomicBloqProblem(bloq))
            return {}

        raise TypeError(f"Unexpected `execute` type: {bloq}.")

    def bloq_in_isa(self, bloq: 'Bloq'):
        import qualtran.bloqs.basic_gates as bg
        from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq
        from qualtran.bloqs.mcmt.and_bloq import And

        # fmt: off
        if isinstance(bloq, (
                bg.TGate, bg.Toffoli, And,
                bg.Rx, bg.Ry, bg.Rz, bg.XPowGate, bg.YPowGate, bg.ZPowGate,
                bg.XGate, bg.YGate, bg.ZGate, bg.Hadamard, bg.SGate,
                bg.CNOT, bg.CYGate, bg.CZ, bg.CHadamard,
                bg.TwoBitSwap, bg.TwoBitCSwap,
                bg.ZeroState, bg.ZeroEffect, bg.OneState, bg.OneEffect,
                bg.PlusState, bg.PlusEffect, bg.MinusState, bg.MinusEffect,
                bg.IntState, bg.IntEffect,
                bg.GlobalPhase, bg.Identity,
                bg.MeasureX, bg.MeasureZ,
                _BookkeepingBloq,
        )):
            return True

        # fmt: on
        return False

    def print_execution_summary(self):
        print(
            f"Simulated execution of {self.n_atoms:,d} ISA operations through {self.n_calls:,d} subroutine calls."
        )
        print("")
        if self.problems:
            problem_count = Counter(self.problems)

            print("During execution, the following problems were encountered:")
            print("")
            for problem, count in problem_count.items():
                print(f" - [{count:4d}x ] {problem.get_summary()}")
        else:
            print("Execution completed successfully.")
