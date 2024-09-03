# CCZ2TFactory
`qualtran.surface_code.CCZ2TFactory`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L27-L157">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Magic state factory costs using the model from catalyzed CCZ to 2T paper.

Inherits From: [`MagicStateFactory`](../../qualtran/surface_code/MagicStateFactory.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.ccz2t_factory.CCZ2TFactory`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.CCZ2TFactory(
    distillation_l1_d=attr_dict[&#x27;distillation_l1_d&#x27;].default,
    distillation_l2_d=attr_dict[&#x27;distillation_l2_d&#x27;].default
)
</code></pre>



<!-- Placeholder for "Used in" -->


<h2 class="add-link">Args</h2>

`distillation_l1_d`<a id="distillation_l1_d"></a>
: Code distance used for level 1 factories.

`distillation_l2_d`<a id="distillation_l2_d"></a>
: Code distance used for level 2 factories.




<h2 class="add-link">References</h2>






<h2 class="add-link">Attributes</h2>

`distillation_l1_d`<a id="distillation_l1_d"></a>
: &nbsp;

`distillation_l2_d`<a id="distillation_l2_d"></a>
: &nbsp;




## Methods

<h3 id="l0_state_injection_error"><code>l0_state_injection_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L47-L53">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l0_state_injection_error(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Error rate associated with the level-0 creation of a |T> state.

By using the techniques of Ying Li (https://arxiv.org/abs/1410.7808), this can be
done with approximately the same error rate as the underlying physical error rate.

<h3 id="l0_topo_error_t_gate"><code>l0_topo_error_t_gate</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L55-L68">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l0_topo_error_t_gate(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Topological error associated with level-0 distillation.

For a level-1 code distance of `d1`, this construction uses a `d1/2` distance code
for storing level-0 T states.

<h3 id="l0_error"><code>l0_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L70-L76">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l0_error(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Chance of failure of a T gate performed with an injected (level-0) T state.

As a simplifying approximation here (and elsewhere) we assume different sources
of error are independent, and we merely add the probabilities.

<h3 id="l1_topo_error_factory"><code>l1_topo_error_factory</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L82-L86">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l1_topo_error_factory(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Topological error associated with a L1 T factory.


<h3 id="l1_topo_error_t_gate"><code>l1_topo_error_t_gate</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L88-L91">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l1_topo_error_t_gate(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>




<h3 id="l1_distillation_error"><code>l1_distillation_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L93-L99">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l1_distillation_error(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

The error due to level-0 faulty T states making it through distillation undetected.

The level 1 distillation procedure detects any two errors. There are 35 weight-three
errors that can make it through undetected.

<h3 id="l1_error"><code>l1_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L101-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l1_error(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Chance of failure of a T gate performed with a T state produced from the L1 factory.


<h3 id="l2_error"><code>l2_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L113-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>l2_error(
    error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Chance of failure of the level two factory.

This is the chance of failure of a CCZ gate or a pair of T gates performed with a CCZ state.

<h3 id="n_physical_qubits"><code>n_physical_qubits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L131-L134">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_physical_qubits() -> int
</code></pre>

The number of physical qubits used by the magic state factory.


<h3 id="factory_error"><code>factory_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L136-L142">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>factory_error(
    n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
) -> float
</code></pre>

Error resulting from the magic state distillation part of the computation.


<h3 id="n_cycles"><code>n_cycles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/ccz2t_factory.py#L144-L157">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_cycles(
    n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
) -> int
</code></pre>

The number of error-correction cycles to distill enough magic states.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class CCZ2TFactory.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class CCZ2TFactory.




