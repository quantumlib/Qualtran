# t_complexity_compat


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/cirq_interop/t_complexity_protocol.py#L200-L225">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the TComplexity of a bloq or some Cirq objects.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.cirq_interop.t_complexity_protocol.t_complexity_compat(
    stc: Any
) -> <a href="../../../qualtran/cirq_interop/t_complexity_protocol/TComplexity.html"><code>qualtran.cirq_interop.t_complexity_protocol.TComplexity</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

The main `t_complexity` function now expects a `Bloq`. Historically, there were strategies
to derive t complexities from other container classes (`cirq.Circuit`, `cirq.Moment`) and
gates/operations.

<h2 class="add-link">Args</h2>

`stc`<a id="stc"></a>
: an object to compute its TComplexity.




<h2 class="add-link">Returns</h2>




<h2 class="add-link">Raises</h2>

`TypeError`<a id="TypeError"></a>
: if the methods fails to compute TComplexity.


