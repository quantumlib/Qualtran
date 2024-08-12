#  Copyright 2024 Google LLC
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

from typing import Tuple

from attr import frozen

from qualtran import QAny, QBit, Register
from qualtran.bloqs.multiplexers.black_box_select import _black_box_select, BlackBoxSelect
from qualtran.bloqs.multiplexers.select_base import SelectOracle


def test_black_box_select(bloq_autotester):
    bloq_autotester(_black_box_select)


@frozen
class TestSelectOracle(SelectOracle):
    @property
    def control_registers(self) -> Tuple[Register, ...]:
        return ()

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (Register('z', QBit()),)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (Register('a', QAny(5)),)


def test_select_oracle():
    bloq = BlackBoxSelect(TestSelectOracle())
    assert bloq.selection_registers == (Register('selection', QAny(1)),)
    assert bloq.target_registers == (Register('system', QAny(5)),)
