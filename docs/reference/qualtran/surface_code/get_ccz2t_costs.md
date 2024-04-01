# get_ccz2t_costs


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_cost_model.py#L164-L194">
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
<p>`qualtran.surface_code.ccz2t_cost_model.get_ccz2t_costs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.get_ccz2t_costs(
    *,
    n_magic: <a href="../../qualtran/surface_code/MagicCount.html"><code>qualtran.surface_code.MagicCount</code></a>,
    n_algo_qubits: int,
    phys_err: float,
    cycle_time_us: float,
    factory: <a href="../../qualtran/surface_code/MagicStateFactory.html"><code>qualtran.surface_code.MagicStateFactory</code></a>,
    data_block: <a href="../../qualtran/surface_code/data_block/DataBlock.html"><code>qualtran.surface_code.data_block.DataBlock</code></a>
) -> <a href="../../qualtran/surface_code/PhysicalCost.html"><code>qualtran.surface_code.PhysicalCost</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Note that this function can return failure probabilities larger than 1.

<h2 class="add-link">Args</h2>

`n_magic`<a id="n_magic"></a>
: The number of magic states (T, Toffoli) required to execute the algorithm

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


