#  Copyright 2023 Google LLC
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

from .atom import TestAtom, TestTwoBitOp, TestGWRAtom
from .casting import TestCastToFrom
from .many_registers import TestMultiRegister
from .with_call_graph import TestBloqWithCallGraph
from .with_decomposition import TestParallelCombo, TestSerialCombo
from .costing import CostingBloq
from .interior_alloc import InteriorAlloc
from .large_bloq import LargeBloq
from .many_registers import TestBoundedQUInt, TestMultiTypedRegister, TestQFxp
from .matrix_gate import MatrixGate
from .qubit_count_many_alloc import TestManyAllocAbstracted, TestManyAllocMany, TestManyAllocOnce
from .random_select_and_prepare import TestPauliSelectOracle, TestPrepareOracle
from .with_decomposition import TestIndependentParallelCombo
