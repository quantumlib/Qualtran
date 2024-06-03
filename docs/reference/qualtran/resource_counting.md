# Module: resource_counting


Counting resource usage (bloqs, qubits)



isort:skip_file
## Modules

[`classify_bloqs`](../qualtran/resource_counting/classify_bloqs.md) module

[`generalizers`](../qualtran/resource_counting/generalizers.md): Bloq counting generalizers.

[`t_counts_from_sigma`](../qualtran/resource_counting/t_counts_from_sigma.md) module

## Classes

[`class SympySymbolAllocator`](../qualtran/resource_counting/SympySymbolAllocator.md): A class that allocates unique sympy symbols for integrating out bloq attributes.

[`class CostKey`](../qualtran/resource_counting/CostKey.md): Abstract base class for different types of costs.

[`class SuccessProb`](../qualtran/resource_counting/SuccessProb.md): The success probability of a bloq.

[`class QubitCount`](../qualtran/resource_counting/QubitCount.md): A cost estimating the number of qubits required to implement a bloq.

[`class BloqCount`](../qualtran/resource_counting/BloqCount.md): A cost which is the count of a specific set of bloqs forming a gateset.

[`class QECGatesCost`](../qualtran/resource_counting/QECGatesCost.md): Counts specifically for 'expensive' gates in a surface code error correction scheme.

[`class GateCounts`](../qualtran/resource_counting/GateCounts.md): A data class of counts of the typical target gates in a compilation.

## Functions

[`big_O(...)`](../qualtran/resource_counting/big_O.md): Helper to deal with CS-style big-O notation that takes the infinite limit by default.

[`build_cbloq_call_graph(...)`](../qualtran/resource_counting/build_cbloq_call_graph.md): Count all the subbloqs in a composite bloq.

[`format_call_graph_debug_text(...)`](../qualtran/resource_counting/format_call_graph_debug_text.md): Print the graph returned from `get_bloq_counts_graph`.

[`get_bloq_call_graph(...)`](../qualtran/resource_counting/get_bloq_call_graph.md): Recursively build the bloq call graph and call totals.

[`get_bloq_callee_counts(...)`](../qualtran/resource_counting/get_bloq_callee_counts.md): Get the direct callees of a bloq and the number of times they are called.

[`get_cost_cache(...)`](../qualtran/resource_counting/get_cost_cache.md): Build a cache of cost values for the bloq and its callees.

[`get_cost_value(...)`](../qualtran/resource_counting/get_cost_value.md): Compute the specified cost of the provided bloq.

[`query_costs(...)`](../qualtran/resource_counting/query_costs.md): Compute a selection of costs for a bloq and its callees.

## Type Aliases

[`BloqCountT`](../qualtran/resource_counting/BloqCountT.md)

[`GeneralizerT`](../qualtran/resource_counting/GeneralizerT.md)



<h2 class="add-link">Other Members</h2>

CostValT<a id="CostValT"></a>
: Instance of `typing.TypeVar`


