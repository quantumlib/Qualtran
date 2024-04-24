# DataBlock
`qualtran.surface_code.data_block.DataBlock`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L25-L48">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A cost model for the data block of a surface code compilation.

<!-- Placeholder for "Used in" -->

A surface code layout is segregated into qubits dedicated to magic state distillation
and others dedicated to storing the actual data being processed. The latter area is
called the data block, and we provide its costs here.

## Methods

<h3 id="footprint"><code>footprint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L33-L40">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>footprint(
    n_algo_qubits: int
) -> int
</code></pre>

The number of physical qubits used by the data block.


Attributes

`n_algo_qubits`
: The number of algorithm qubits whose data must be stored and
  accessed.




<h3 id="data_error"><code>data_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L42-L44">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>data_error(
    n_algo_qubits: int, n_cycles: int, phys_err: float
) -> float
</code></pre>

The error associated with storing data on `n_algo_qubits` for `n_cycles`.


<h3 id="n_cycles_to_consume_a_magic_state"><code>n_cycles_to_consume_a_magic_state</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L46-L48">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>n_cycles_to_consume_a_magic_state() -> int
</code></pre>

The worst case number of cycles needed to consume a magic state.




