# get_ccz2t_costs_from_grid_search


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/gidney_fowler_model.py#L182-L238">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Grid search over parameters to minimize the space-time volume.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.gidney_fowler_model.get_ccz2t_costs_from_grid_search`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.get_ccz2t_costs_from_grid_search(
    *,
    n_logical_gates: 'GateCounts',
    n_algo_qubits: int,
    phys_err: float = 0.001,
    error_budget: float = 0.01,
    cycle_time_us: float = 1.0,
    factory_iter: Iterable[<a href="../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>] = tuple(iter_ccz2t_factories()),
    data_block_iter: Iterable[<a href="../../qualtran/surface_code/DataBlock.html"><code>qualtran.surface_code.DataBlock</code></a>] = tuple(iter_simple_data_blocks()),
    cost_function: Callable[[<a href="../../qualtran/surface_code/PhysicalCostsSummary.html"><code>qualtran.surface_code.PhysicalCostsSummary</code></a>], float] = (lambda pc: pc.qubit_hours)
) -> Tuple[<a href="../../qualtran/surface_code/PhysicalCostsSummary.html"><code>qualtran.surface_code.PhysicalCostsSummary</code></a>, <a href="../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>, <a href="../../qualtran/surface_code/SimpleDataBlock.html"><code>qualtran.surface_code.SimpleDataBlock</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`n_logical_gates`<a id="n_logical_gates"></a>
: Number of algorithm logical gates.

`n_algo_qubits`<a id="n_algo_qubits"></a>
: Number of algorithm logical qubits.

`phys_err`<a id="phys_err"></a>
: The physical error rate of the device. This sets the suppression
  factor for increasing code distance.

`error_budget`<a id="error_budget"></a>
: The acceptable chance of an error occurring at any point. This includes
  data storage failures as well as top-level distillation failure.

`cycle_time_us`<a id="cycle_time_us"></a>
: The number of microseconds it takes to execute a surface code cycle.

`factory_iter`<a id="factory_iter"></a>
: iterable containing the instances of MagicStateFactory to search over.

`data_block_iter`<a id="data_block_iter"></a>
: iterable containing the instances of SimpleDataBlock to search over.

`cost_function`<a id="cost_function"></a>
: function of PhysicalCostsSummary to be minimized. Defaults to spacetime volume.
  Set `cost_function = (lambda pc: pc.duration_hr)` to mimimize wall time.




<h2 class="add-link">Returns</h2>




<h2 class="add-link">References</h2>


