# Module: resource_counting


Counting resource usage (bloqs, qubits)



isort:skip_file
## Modules

[`bloq_counts`](../qualtran/resource_counting/bloq_counts.md): Functionality for the <a href="../qualtran/Bloq.html#bloq_counts"><code>Bloq.bloq_counts()</code></a> protocol.

## Classes

[`class SympySymbolAllocator`](../qualtran/resource_counting/SympySymbolAllocator.md): A class that allocates unique sympy symbols for integrating out bloq attributes.

[`class GraphvizCounts`](../qualtran/resource_counting/GraphvizCounts.md): This class turns a bloqs count graph into Graphviz objects and drawings.

## Functions

[`big_O(...)`](../qualtran/resource_counting/big_O.md): Helper to deal with CS-style big-O notation that takes the infinite limit by default.

[`get_bloq_counts_graph(...)`](../qualtran/resource_counting/get_bloq_counts_graph.md): Recursively gather bloq counts.

[`get_cbloq_bloq_counts(...)`](../qualtran/resource_counting/get_cbloq_bloq_counts.md): Count all the subbloqs in a composite bloq.

[`markdown_bloq_expr(...)`](../qualtran/resource_counting/markdown_bloq_expr.md): Return "`bloq`: expr" as markdown.

[`markdown_counts_graph(...)`](../qualtran/resource_counting/markdown_counts_graph.md): Render the graph returned from `get_bloq_counts_graph` as markdown.

[`markdown_counts_sigma(...)`](../qualtran/resource_counting/markdown_counts_sigma.md)

[`print_counts_graph(...)`](../qualtran/resource_counting/print_counts_graph.md): Print the graph returned from `get_bloq_counts_graph`.

## Type Aliases

[`BloqCountT`](../qualtran/resource_counting/BloqCountT.md)

