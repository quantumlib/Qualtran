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

from qualtran import BloqBuilder
from qualtran.bloqs.arithmetic import SignedIntegerToTwosComplement, ToContiguousIndex
from qualtran.bloqs.basic_gates import TGate


def _make_to_contiguous_index():
    from qualtran.bloqs.arithmetic import ToContiguousIndex

    return ToContiguousIndex(bitsize=4, s_bitsize=8)


def _make_signed_to_twos_complement():
    from qualtran.bloqs.arithmetic import SignedIntegerToTwosComplement

    return SignedIntegerToTwosComplement(bitsize=10)


def test_to_contiguous_index():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('mu', bitsize)
    q1 = bb.add_register('nu', bitsize)
    out = bb.add_register('s', 1)
    q0, q1, out = bb.add(ToContiguousIndex(bitsize, 2 * bitsize), mu=q0, nu=q1, s=out)
    cbloq = bb.finalize(mu=q0, nu=q1, s=out)
    cbloq.t_complexity()


def test_signed_to_twos_complement():
    bb = BloqBuilder()
    bitsize = 5
    q0 = bb.add_register('x', bitsize)
    q0 = bb.add(SignedIntegerToTwosComplement(bitsize), x=q0)
    cbloq = bb.finalize(x=q0)
    _, sigma = cbloq.call_graph()
    assert sigma[TGate()] == 4 * (5 - 2)
