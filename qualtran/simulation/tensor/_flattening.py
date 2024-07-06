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

from qualtran import Bloq, CompositeBloq


def bloq_has_custom_tensors(bloq: Bloq) -> bool:
    """Whether this bloq declares custom tensors by overriding `.my_tensors(...)`.

    This is a heuristic that checks that the method is overriden. This is used as
    an optional predicate in `flatten_for_tensor_contraction`.
    """
    return not bloq.my_tensors.__qualname__.startswith(
        'Bloq.'
    ) and not bloq.my_tensors.__qualname__.startswith('GateWithRegisters.')


def flatten_for_tensor_contraction(bloq: Bloq, full_flatten: bool = True) -> CompositeBloq:
    """Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.

    Args:
        bloq: The bloq to flatten.
        full_flatten: Whether to completely flatten the bloq into the smallest possible
            bloqs. Otherwise, stop flattening if custom tensors are encountered.
    """
    cbloq = bloq.as_composite_bloq()
    if full_flatten:
        pred = lambda b: True
    else:
        pred = lambda binst: not bloq_has_custom_tensors(binst.bloq)

    return cbloq.flatten(pred)
