# get_ccz2t_costs


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/gidney_fowler_model.py#L32-L68">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generate spacetime cost and failure probability given physical and logical parameters.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.gidney_fowler_model.get_ccz2t_costs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.get_ccz2t_costs(
    *,
    n_logical_gates: 'GateCounts',
    n_algo_qubits: int,
    phys_err: float,
    cycle_time_us: float,
    factory: <a href="../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>,
    data_block: <a href="../../qualtran/surface_code/DataBlock.html"><code>qualtran.surface_code.DataBlock</code></a>
) -> <a href="../../qualtran/surface_code/PhysicalCostsSummary.html"><code>qualtran.surface_code.PhysicalCostsSummary</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Note that this function can return failure probabilities larger than 1.

This function exists for backwards-compatibility. Consider constructing a `PhysicalCostModel`
directly.

<h2 class="add-link">Args</h2>

`n_logical_gates`<a id="n_logical_gates"></a>
: The number of algorithm logical gates.

`n_algo_qubits`<a id="n_algo_qubits"></a>
: Number of algorithm logical qubits.

`phys_err`<a id="phys_err"></a>
: The physical error rate of the device.

`cycle_time_us`<a id="cycle_time_us"></a>
: The number of microseconds it takes to execute a surface code cycle.

`factory`<a id="factory"></a>
: magic state factory configuration. Used to evaluate distillation error and cost.

`data_block`<a id="data_block"></a>
: data block configuration. Used to evaluate data error and footprint.


