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
from functools import cached_property

from attrs import frozen

from qualtran import Bloq, QDType, Signature
from qualtran.bloqs.arithmetic import AddK
from qualtran.bloqs.basic_gates import OnEach, XGate


@frozen
class Negate(Bloq):
    """Compute the two's complement for a signed integer/fixed-point value.

    The two's complement is computed by the following steps:
    1. Flip all the bits
    2. Add 1 to the value (interpreted as an unsigned integer), ignoring the overflow.

    For a controlled negate bloq: the second step uses a quantum-quantum adder by
    loading the constant (i.e. 1), therefore has an improved controlled version
    which only controls the constant load and not the adder circuit, hence halving
    the T-cost compared to a controlled adder.

    Args:
        dtype: The data type of the input value.

    Registers:
        x: Any signed value stored in two's complement form.
    """

    dtype: QDType

    @cached_property
    def signature(self) -> 'Signature':
        return Signature.build_from_dtypes(x=self.dtype)

    def build_composite_bloq(self, bb: 'BloqBuilder', x: 'SoquetT') -> dict[str, 'SoquetT']:
        x = bb.add(OnEach(self.dtype.num_qubits, XGate()), q=x)
        x = bb.add(AddK(self.dtype.num_qubits, k=1), x=x)
        return {'x': x}
