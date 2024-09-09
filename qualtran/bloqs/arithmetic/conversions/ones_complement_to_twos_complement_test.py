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
from qualtran import BloqBuilder
from qualtran.bloqs.arithmetic.conversions.ones_complement_to_twos_complement import (
    _signed_to_twos,
    SignedIntegerToTwosComplement,
)
from qualtran.bloqs.basic_gates import Toffoli


def test_signed_to_twos(bloq_autotester):
    bloq_autotester(_signed_to_twos)


def test_signed_to_twos_complement_toffoli_count():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    q0 = bb.add(SignedIntegerToTwosComplement(bitsize), x=q0)
    cbloq = bb.finalize(x=q0)
    _, sigma = cbloq.call_graph()
    assert sigma[Toffoli()] == (5 - 2)
