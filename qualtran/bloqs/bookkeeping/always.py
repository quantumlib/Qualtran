#  Copyright 2025 Google LLC
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
from typing import Iterable, Optional, Sequence

import attrs

from qualtran import (
    AddControlledT,
    Bloq,
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    CtrlSpec,
    Signature,
    SoquetT,
)


@attrs.frozen
class Always(Bloq):
    """Always execute the wrapped bloq, even when a controlled version is requested

    A controlled version of a composite bloq in turn controls each subbloq in the decomposition.
    Wrapping a particular subbloq with `Always` lets it bypass the controls,
    i.e. it is "always" executed, irrespective of what the controls are.

    This is useful when writing decompositions for two known patterns:

    1. Compute-uncompute pairs: If a decomposition contains a compute-uncompute pair,
    then for a controlled version, we only need to control the rest of the bloqs.
    Wrapping both the compute and uncompute bloqs in `Always` lets them bypass the controls.

    2. Controlled data-loading: For example, in the `AddK` bloq which adds a constant `k` to the
    register, we (controlled) load the value `k` into a quantum register, and "always" perform an
    quantum-quantum addition using `Add`, and unload `k`. Here wrapping the middle `Add` with
    `Always` lets it bypass controls, e.g. when using `AddK.controlled()`.

    This simplifies the decompositions by avoiding the need to explicitly define the decomposition
    for the controlled version of bloq.

    **Caution:** This wrapper should be used with care. It is up to the bloq author to ensure that
    the controlled version of a decomposition containing `Always` bloqs still respects the
    controlled protocol. That is, ignoring controls on these subbloqs wrapped in `Always` should not
    change the action of the overall bloq with respect to the reference controlled implementation.

    Args:
        subbloq: The bloq to always apply, irrespective of any controls.
    """

    subbloq: Bloq

    @property
    def signature(self) -> 'Signature':
        return self.subbloq.signature

    def build_composite_bloq(self, bb: 'BloqBuilder', **soqs: 'SoquetT') -> dict[str, 'SoquetT']:
        return bb.add_d(self.subbloq, **soqs)

    def get_ctrl_system(
        self, ctrl_spec: Optional['CtrlSpec'] = None
    ) -> tuple['Bloq', 'AddControlledT']:
        """Pass-through the control registers as-is"""

        def add_controlled(
            bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
        ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
            out_soqs = bb.add_t(self, **in_soqs)
            return ctrl_soqs, out_soqs

        return self, add_controlled

    def adjoint(self) -> 'Always':
        return Always(self.subbloq.adjoint())

    def __str__(self) -> str:
        return str(self.subbloq)


@bloq_example
def _always_and() -> Always:
    from qualtran.bloqs.mcmt.and_bloq import And

    always_and = Always(And())

    return always_and


_ALWAYS_DOC = BloqDocSpec(bloq_cls=Always, examples=[_always_and])
