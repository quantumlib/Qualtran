# MagicStateFactory
`qualtran.surface_code.MagicStateFactory`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L20-L38">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A cost model for the magic state distillation factory of a surface code compilation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.magic_state_factory.MagicStateFactory`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

A surface code layout is segregated into qubits dedicated to magic state distillation
and storing the data being processed. The former area is called the magic state distillation
factory, and we provide its costs here.

## Methods

<h3 id="footprint"><code>footprint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L28-L30">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>footprint() -> int
</code></pre>

The number of physical qubits used by the magic state factory.


<h3 id="n_cycles"><code>n_cycles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L32-L34">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>n_cycles(
    n_magic: <a href="../../qualtran/surface_code/MagicCount.html"><code>qualtran.surface_code.MagicCount</code></a>,
    phys_err: float
) -> int
</code></pre>

The number of cycles (time) required to produce the requested number of magic states.


<h3 id="distillation_error"><code>distillation_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/magic_state_factory.py#L36-L38">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>distillation_error(
    n_magic: <a href="../../qualtran/surface_code/MagicCount.html"><code>qualtran.surface_code.MagicCount</code></a>,
    phys_err: float
) -> float
</code></pre>

The total error expected from distilling magic states with a given physical error rate.




