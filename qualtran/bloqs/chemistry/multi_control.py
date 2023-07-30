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
from functools import cached_property
from typing import Tuple

from attrs import frozen

from qualtran import Bloq, Register, Signature


@frozen
class MultiControl(Bloq):
    """A multi-controlled gate.

    Args:
        cvs: A tuple of control variable settings. Each entry specifies whether that
            control line is a "positive" control (`cv[i]=1`) or a "negative" control `0`.

    Registers:
     - ctrl: An n-bit control register.
    """

    bitsizes: Tuple[int, ...]
    cvs: Tuple[int, ...]

    @cached_property
    def signature(self) -> Signature:
        return Signature([Register(f'ctrl_{i}', bitsize=bs) for i, bs in enumerate(self.bitsizes)])
