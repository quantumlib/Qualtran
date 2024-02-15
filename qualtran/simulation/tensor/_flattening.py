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
    """Whether this bloq declares custom tensors by overriding `.add_my_tensors(...)`.

    This is a heuristic that checks that the method is overriden. This is used as the
    flattening predicate in `flatten_for_tensor_contraction`.
    """
    return not bloq.add_my_tensors.__qualname__.startswith(
        'Bloq.'
    ) and not bloq.add_my_tensors.__qualname__.startswith('GateWithRegisters.')


def flatten_for_tensor_contraction(bloq: Bloq, max_depth: int = 1_000) -> CompositeBloq:
    """Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.

    Without this function, bloqs without custom tensors will be contracted to a dense tensor using
    their decomposition and then that dense tensor will be used in the enclosing tensor network.
    To allow a more efficient contraction ordering, use this function to decompose-and-flatten
    as much as possible before starting the tensor contraction.
    """
    cbloq = bloq.as_composite_bloq()
    return cbloq.flatten(lambda binst: not bloq_has_custom_tensors(binst.bloq), max_depth=max_depth)
