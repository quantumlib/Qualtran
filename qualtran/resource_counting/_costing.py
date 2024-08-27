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

import abc
import collections
import logging
import time
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from qualtran import CompositeBloq

from ._generalization import _make_composite_generalizer, GeneralizerT

if TYPE_CHECKING:
    from qualtran import Bloq

logger = logging.getLogger(__name__)

CostValT = TypeVar('CostValT')


class CostKey(Generic[CostValT], metaclass=abc.ABCMeta):
    """Abstract base class for different types of costs.

    One important aspect of a bloq is the resources required to execute it on an error
    corrected quantum computer. Since we're usually trying to minimize these resource requirements
    we will generally use the catch-all term "costs".

    There are a variety of different types or flavors of costs. Each is represented by an
    instance of a sublcass of `CostKey`. For example, gate counts (including T-gate counts),
    qubit requirements, and circuit depth are all cost metrics that may be of interest.

    Each `CostKey` primarily encodes the behavior required to compute a cost value from a
    bloq. Often, these costs are defined recursively: a bloq's costs is some combination
    of the costs of the bloqs in its decomposition (i.e. the bloq 'callees'). Implementors
    must override the `compute` method to define the cost computation.

    Each cost key has an associated CostValT. For example, the CostValT of a "t count"
    CostKey could be an integer. For a more complicated gateset, the value could be a mapping
    from gate to count. This abstract base class is generic w.r.t. `CostValT`. Subclasses
    should have a concrete value type. The `validate_val` method can optionally be overridden
    to raise an exception if a bad value type is encountered. The `zero` method must return
    the zero (additive identity) cost value of the correct type.
    """

    @abc.abstractmethod
    def compute(self, bloq: 'Bloq', get_callee_cost: Callable[['Bloq'], CostValT]) -> CostValT:
        """Compute this type of cost.

        When implementing a new CostKey, this method must be overridden.
        Users should not call this method directly. Instead: use the `qualtran.resource_counting`
        functions like `get_cost_value`, `get_cost_cache`, or `query_costs`. These provide
        caching, logging, generalizers, and support for static costs.

        For recursive computations, use the provided callable to recurse.

        Args:
            bloq: The bloq to compute the cost of.
            get_callee_cost: A qualtran-provided function for computing costs for "callees"
                of the bloq; i.e. bloqs in the decomposition. Use this function to accurately
                cache intermediate cost values and respect bloqs' static costs.

        Returns:
            A value of the generic type `CostValT`. Subclasses should define their value type.
        """

    @abc.abstractmethod
    def zero(self) -> CostValT:
        """The value corresponding to zero cost."""

    def validate_val(self, val: CostValT):
        """Assert that `val` is a valid `CostValT`.

        This method can be optionally overridden to raise an error if an invalid value
        is encountered. By default, no validation is performed.
        """


def _get_cost_value(
    bloq: 'Bloq',
    cost_key: CostKey[CostValT],
    *,
    costs_cache: Dict['Bloq', CostValT],
    generalizer: 'GeneralizerT',
) -> CostValT:
    """Helper function for getting costs.

    This function tries the following strategies
     1. Use the value found in `costs_cache`, if it exists.
     2. Use the value returned by `Bloq.my_static_costs` if one is returned.
     3. Use `cost_key.compute()` and cache the result in `costs_cache`.

    Args:
        bloq: The bloq.
        cost_key: The cost key to get the value for.
        costs_cache: A dictionary to use as a cache for computed bloq costs. This cache
            will be mutated by this function.
        generalizer: The generalizer to operate on each bloq before computing its cost.
    """
    bloq = generalizer(bloq)
    if bloq is None:
        return cost_key.zero()

    # Strategy 1: Use cached value
    if not isinstance(bloq, CompositeBloq) and bloq in costs_cache:
        logger.debug("Using cached %s for %s", cost_key, bloq)
        return costs_cache[bloq]

    # Strategy 2: Static costs
    static_cost = bloq.my_static_costs(cost_key)
    if static_cost is not NotImplemented:
        cost_key.validate_val(static_cost)
        logger.info("Using static %s for %s", cost_key, bloq)
        costs_cache[bloq] = static_cost
        return static_cost

    # Strategy 3: Compute
    # part a. set up caching of computed costs by currying the costs_cache.
    def _get_cost_val_internal(callee: 'Bloq'):
        return _get_cost_value(callee, cost_key, costs_cache=costs_cache, generalizer=generalizer)

    # part b. call the compute method and cache the result.
    tstart = time.perf_counter()
    computed_cost = cost_key.compute(bloq, _get_cost_val_internal)
    tdur = time.perf_counter() - tstart
    logger.info("Computed %s for %s in %g s", cost_key, bloq, tdur)
    if not isinstance(bloq, CompositeBloq):
        costs_cache[bloq] = computed_cost
    return computed_cost


def get_cost_value(
    bloq: 'Bloq',
    cost_key: CostKey[CostValT],
    costs_cache: Optional[Dict['Bloq', CostValT]] = None,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
) -> CostValT:
    """Compute the specified cost of the provided bloq.

    Args:
        bloq: The bloq to compute the cost of.
        cost_key: A CostKey that specifies which cost to compute.
        costs_cache: If provided, use this dictionary of cached cost values. Values in this
            dictionary will be preferred over computed values (even if they disagree). This
            dictionary will be mutated by the function.
        generalizer: If provided, run this function on each bloq in the call graph to dynamically
            modify attributes. If the function returns `None`, the bloq is ignored in the
            cost computation. If a sequence of generalizers is provided, each generalizer
            will be run in order.

    Returns:
        The cost value. Its type depends on the provided `cost_key`.
    """
    if costs_cache is None:
        costs_cache = {}
    if generalizer is None:
        generalizer = lambda b: b
    if isinstance(generalizer, collections.abc.Sequence):
        generalizer = _make_composite_generalizer(*generalizer)

    cost_val = _get_cost_value(bloq, cost_key, costs_cache=costs_cache, generalizer=generalizer)
    return cost_val


def get_cost_cache(
    bloq: 'Bloq',
    cost_key: CostKey[CostValT],
    costs_cache: Optional[Dict['Bloq', CostValT]] = None,
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
) -> Dict['Bloq', CostValT]:
    """Build a cache of cost values for the bloq and its callees.

    This can be useful to inspect how callees' costs flow upwards in a given cost computation.

    Args:
        bloq: The bloq to seed the cost computation.
        cost_key: A CostKey that specifies which cost to compute.
        costs_cache: If provided, use this dictionary for initial cached cost values. Values in this
            dictionary will be preferred over computed values (even if they disagree). This
            dictionary will be mutated by the function. This dictionary will be returned by the
            function.
        generalizer: If provided, run this function on each bloq in the call graph to dynamically
            modify attributes. If the function returns `None`, the bloq is ignored in the
            cost computation. If a sequence of generalizers is provided, each generalizer
            will be run in order.

    Returns:
        A dictionary mapping bloqs to cost values. The value type depends on the `cost_key`.
        The bloqs in the mapping depend on the recursive nature of the cost key.
    """
    if costs_cache is None:
        costs_cache = {}
    if generalizer is None:
        generalizer = lambda b: b
    if isinstance(generalizer, collections.abc.Sequence):
        generalizer = _make_composite_generalizer(*generalizer)

    _get_cost_value(bloq, cost_key, costs_cache=costs_cache, generalizer=generalizer)
    return costs_cache


def query_costs(
    bloq: 'Bloq',
    cost_keys: Iterable[CostKey],
    generalizer: Optional[Union['GeneralizerT', Sequence['GeneralizerT']]] = None,
) -> Dict['Bloq', Dict[CostKey, CostValT]]:
    """Compute a selection of costs for a bloq and its callees.

    This function can be used to annotate a call graph diagram with multiple costs
    for each bloq. Specifically, the return value of this function can be used as the
    `bloq_data` argument to `GraphvizCallGraph`.

    Args:
        bloq: The bloq to seed the cost computation.
        cost_keys: A sequence of CostKey that specifies which costs to compute.
        generalizer: If provided, run this function on each bloq in the call graph to dynamically
            modify attributes. If the function returns `None`, the bloq is ignored in the
            cost computation. If a sequence of generalizers is provided, each generalizer
            will be run in order.

    Returns:
        A dictionary of dictionaries forming a table of multiple costs for multiple bloqs.
        This is indexed by bloq, then cost key.
    """
    costs: Dict['Bloq', Dict[CostKey, CostValT]] = defaultdict(dict)
    for cost_key in cost_keys:
        cost_for_bloqs = get_cost_cache(bloq, cost_key, generalizer=generalizer)
        for bloq, val in cost_for_bloqs.items():
            costs[bloq][cost_key] = val
    return dict(costs)
