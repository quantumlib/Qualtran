# MagicStateFactory
`qualtran.surface_code.MagicStateFactory`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L23-L61">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Methods for modeling the costs of the magic state factories of a surface code compilation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.magic_state_factory.MagicStateFactory`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

An important consideration for a surface code compilation is how to execute arbitrary gates
to run the desired algorithm. The surface code can execute Clifford gates in a fault-tolerant
manner. Non-Clifford gates like the T gate, Toffoli or CCZ gate, or non-Clifford rotation
gates require more expensive gadgets to implement. Executing a T or CCZ gate requires first
using the technique of state distillation in an area of the computation called a "magic state
factory" to distill a noisy T or CCZ state into a "magic state" of sufficiently low error.
Such quantum states can be used to enact the non-Clifford quantum gate through gate
teleportation.

Magic state production is thought to be an important runtime and qubit-count bottleneck in
foreseeable fault-tolerant quantum computers.

This abstract interface specifies that each magic state factory must report its required
number of physical qubits, the number of error correction cycles to produce enough magic
states to enact a given number of logical gates and an error model, and the expected error
associated with generating those magic states.

## Methods

<h3 id="n_physical_qubits"><code>n_physical_qubits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L44-L46">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>n_physical_qubits() -> int
</code></pre>

The number of physical qubits used by the magic state factory.


<h3 id="n_cycles"><code>n_cycles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L48-L52">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>n_cycles(
    n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
) -> int
</code></pre>

The number of cycles (time) required to produce the requested number of magic states.


<h3 id="factory_error"><code>factory_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L54-L61">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>factory_error(
    n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
) -> float
</code></pre>

The total error expected from distilling magic states with a given physical error rate.

This includes the cumulative effects of data-processing errors and distillation failures.



