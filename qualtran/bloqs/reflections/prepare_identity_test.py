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
from qualtran import Signature
from qualtran.bloqs.basic_gates import Identity
from qualtran.bloqs.reflections.prepare_identity import _prepare_identity, PrepareIdentity


def test_prepare_identity(bloq_autotester):
    bloq_autotester(_prepare_identity)


def test_prepare_identity_call_graph():
    bloq = PrepareIdentity(tuple(Signature.build(a=4, b=4, c=5)))
    _, sigma = bloq.call_graph()
    assert sigma == {Identity(4): 2, Identity(5): 1}
