# PhysicalCostModel
`qualtran.surface_code.PhysicalCostModel`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L31-L188">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A model for estimating physical costs from algorithm counts.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.physical_cost_model.PhysicalCostModel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.PhysicalCostModel(
    physical_params, data_block, factory, qec_scheme
)
</code></pre>



<!-- Placeholder for "Used in" -->

The model is parameterized by 1) properties of the target hardware architecture encapsulated
in the data class `PhysicalParameters`, and 2) Execution protocol design choices.

We further factor the design choices into a) the data block design for storing
algorithm qubits, b) the magic state factory construction, and c) the error suppression
ability of the code.

Each method for computing physical costs take `AlgorithmSummary` inputs: the number of
algorithm qubits and the number of algorithm gates. Output quantities
include the wall-clock time, the number of physical qubits, and the probability of failure
due to the physical realization of the algorithm.

### Time costs

The amount of time to run an algorithm is modeled as the greater of two quantities:
The number of cycles required to generate enough magic states (via the `factory`), and
the number of cycles required to consume the magic states (via the `data_block`). The model
assumes that the rate of magic state generation is slower than the reaction limit. Each
cycle takes a fixed amount of wall-clock time, given by `architecture`.

### Space costs

The number of physical qubits is the sum of the number of factory qubits and data block qubits.

### Error

We assume the constituent error probabilities are sufficiently low to permit a first-order
approximation for combining sources of error. The total error is the sum of error probabilities
due to magic state production (via `factory`) and data errors (via `data_block`). Note that
the total error in data storage depends on the number of cycles, which depends on the
factory design.

<h2 class="add-link">Args</h2>

`physical_params`<a id="physical_params"></a>
: The physical parameters of the target hardware

`data_block`<a id="data_block"></a>
: The design of the data block

`factory`<a id="factory"></a>
: The construction of the magic state factory/ies

`qec_scheme`<a id="qec_scheme"></a>
: The scheme used to suppress errors.






<h2 class="add-link">Attributes</h2>

`data_block`<a id="data_block"></a>
: &nbsp;

`factory`<a id="factory"></a>
: &nbsp;

`logical_error_model`<a id="logical_error_model"></a>
: &nbsp;

`physical_params`<a id="physical_params"></a>
: &nbsp;

`qec_scheme`<a id="qec_scheme"></a>
: &nbsp;




## Methods

<h3 id="n_cycles"><code>n_cycles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L122-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_cycles(
    algo_summary: 'AlgorithmSummary'
) -> int
</code></pre>

The number of error correction cycles required to execute the algorithm.


<h3 id="duration_hr"><code>duration_hr</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L127-L132">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_hr(
    algo_summary: 'AlgorithmSummary'
)
</code></pre>

The duration in hours required to execute the algorithm.


<h3 id="n_phys_qubits"><code>n_phys_qubits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L134-L137">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_phys_qubits(
    algo_summary: 'AlgorithmSummary'
) -> int
</code></pre>

The number of physical qubits required to execute the algorithm


<h3 id="error"><code>error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L139-L142">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>error(
    algo_summary: 'AlgorithmSummary'
) -> float
</code></pre>

The total error rate of executing the algorithm.


<h3 id="make_gidney_fowler"><code>make_gidney_fowler</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L144-L158">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>make_gidney_fowler(
    data_d: int
)
</code></pre>




<h3 id="make_beverland_et_al"><code>make_beverland_et_al</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/physical_cost_model.py#L160-L188">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>make_beverland_et_al(
    data_d: int,
    data_block_name: str = &#x27;compact&#x27;,
    factory_ds: Tuple = (9, 3, 3)
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class PhysicalCostModel.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class PhysicalCostModel.




