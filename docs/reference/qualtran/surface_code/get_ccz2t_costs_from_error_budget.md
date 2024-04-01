# get_ccz2t_costs_from_error_budget


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_cost_model.py#L197-L267">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Physical costs using the model from catalyzed CCZ to 2T paper.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.ccz2t_cost_model.get_ccz2t_costs_from_error_budget`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.get_ccz2t_costs_from_error_budget(
    *,
    n_magic: <a href="../../qualtran/surface_code/MagicCount.html"><code>qualtran.surface_code.MagicCount</code></a>,
    n_algo_qubits: int,
    phys_err: float = 0.001,
    error_budget: float = 0.01,
    cycle_time_us: float = 1.0,
    routing_overhead: float = 0.5,
    factory: Optional[<a href="../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>] = None,
    data_block: Optional[<a href="../../qualtran/surface_code/data_block/DataBlock.html"><code>qualtran.surface_code.data_block.DataBlock</code></a>] = None
) -> <a href="../../qualtran/surface_code/PhysicalCost.html"><code>qualtran.surface_code.PhysicalCost</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`n_magic`<a id="n_magic"></a>
: The number of magic states (T, Toffoli) required to execute the algorithm

`n_algo_qubits`<a id="n_algo_qubits"></a>
: Number of algorithm logical qubits.

`phys_err`<a id="phys_err"></a>
: The physical error rate of the device. This sets the suppression
  factor for increasing code distance.

`error_budget`<a id="error_budget"></a>
: The acceptable chance of an error occurring at any point. This includes
  data storage failures as well as top-level distillation failure. By default,
  this follows the prescription of the paper: distillation error is fixed by
  factory parameters and `n_magic`. The data block code distance is then chosen
  from the remaining error budget. If distillation error exceeds the budget, the cost
  estimate will fail. If the `data_block` argument is provided, this argument is
  ignored.

`cycle_time_us`<a id="cycle_time_us"></a>
: The number of microseconds it takes to execute a surface code cycle.

`routing_overhead`<a id="routing_overhead"></a>
: Additional space needed for moving magic states and data qubits around
  in order to perform operations. If the `data_block` argument is provided, this
  argument is ignored.

`factory`<a id="factory"></a>
: By default, construct a default `CCZ2TFactory()`. Otherwise, you can provide
  your own factory or factory configuration using this argument.

`data_block`<a id="data_block"></a>
: By default, construct a `SimpleDataBlock()` according to the `error_budget`.
  Otherwise, provide your own data block.




<h2 class="add-link">References</h2>


