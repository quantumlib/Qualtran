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
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qualtran import Bloq

GeneralizerT = Callable[['Bloq'], Optional['Bloq']]


def _make_composite_generalizer(*funcs: 'GeneralizerT') -> 'GeneralizerT':
    """Return a generalizer that calls each `*funcs` generalizers in order."""

    def _composite_generalize(b: Optional['Bloq']) -> Optional['Bloq']:
        for func in funcs:
            if b is None:
                return None
            b = func(b)
        return b

    return _composite_generalize
