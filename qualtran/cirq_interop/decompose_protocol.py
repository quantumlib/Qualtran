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

from typing import Any

import cirq
from cirq.protocols.decompose_protocol import DecomposeResult


def _decompose_once_considering_known_decomposition(val: Any) -> DecomposeResult:
    """Decomposes a value into operations, if possible.

    Args:
        val: The value to decompose into operations.

    Returns:
        A tuple of operations if decomposition succeeds.
    """
    import uuid

    context = cirq.DecompositionContext(
        qubit_manager=cirq.GreedyQubitManager(prefix=f'_{uuid.uuid4()}', maximize_reuse=True)
    )

    if isinstance(val, cirq.Gate):
        decomposed = cirq.decompose_once_with_qubits(
            val, cirq.LineQid.for_gate(val), context=context, flatten=False, default=None
        )
    else:
        decomposed = cirq.decompose_once(val, context=context, flatten=False, default=None)

    return [*cirq.flatten_to_ops(decomposed)] if decomposed is not None else None
