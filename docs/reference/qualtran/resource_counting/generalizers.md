# Module: generalizers


Bloq counting generalizers.



`Bloq.get_bloq_counts_graph(...)` takes a `generalizer` argument which can combine
multiple bloqs whose attributes differ in ways that do not affect the cost estimates
into one, more general bloq. The functions in this module can be used as generalizers
for this argument.
## Functions

[`cirq_to_bloqs(...)`](../../qualtran/resource_counting/generalizers/cirq_to_bloqs.md): A generalizer that replaces Cirq gates with their equivalent bloq, where possible.

[`generalize_cvs(...)`](../../qualtran/resource_counting/generalizers/generalize_cvs.md): A generalizer that replaces control variables with a shared symbol.

[`generalize_rotation_angle(...)`](../../qualtran/resource_counting/generalizers/generalize_rotation_angle.md): A generalizer that replaces rotation angles with a shared symbol.

[`ignore_alloc_free(...)`](../../qualtran/resource_counting/generalizers/ignore_alloc_free.md): A generalizer that ignores allocations and frees.

[`ignore_cliffords(...)`](../../qualtran/resource_counting/generalizers/ignore_cliffords.md): A generalizer that ignores known clifford bloqs.

[`ignore_split_join(...)`](../../qualtran/resource_counting/generalizers/ignore_split_join.md): A generalizer that ignores split and join operations.



<h2 class="add-link">Other Members</h2>

CV<a id="CV"></a>
: Instance of `sympy.core.symbol.Symbol`

PHI<a id="PHI"></a>
: Instance of `sympy.core.symbol.Symbol`


