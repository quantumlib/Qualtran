# GateCounts
`qualtran.resource_counting.GateCounts`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L123-L235">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A data class of counts of the typical target gates in a compilation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.resource_counting.GateCounts(
    *,
    t=attr_dict[&#x27;t&#x27;].default,
    toffoli=attr_dict[&#x27;toffoli&#x27;].default,
    cswap=attr_dict[&#x27;cswap&#x27;].default,
    and_bloq=attr_dict[&#x27;and_bloq&#x27;].default,
    clifford=attr_dict[&#x27;clifford&#x27;].default,
    rotation=attr_dict[&#x27;rotation&#x27;].default,
    measurement=attr_dict[&#x27;measurement&#x27;].default
)
</code></pre>



<!-- Placeholder for "Used in" -->

Specifically, this class holds counts for the number of `TGate` (and adjoint), `Toffoli`,
`TwoBitCSwap`, `And`, clifford bloqs, single qubit rotations, and measurements.



<h2 class="add-link">Attributes</h2>

`and_bloq`<a id="and_bloq"></a>
: &nbsp;

`clifford`<a id="clifford"></a>
: &nbsp;

`cswap`<a id="cswap"></a>
: &nbsp;

`measurement`<a id="measurement"></a>
: &nbsp;

`rotation`<a id="rotation"></a>
: &nbsp;

`t`<a id="t"></a>
: &nbsp;

`toffoli`<a id="toffoli"></a>
: &nbsp;




## Methods

<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L139-L151">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L153-L162">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L164-L165">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other
)
</code></pre>




<h3 id="asdict"><code>asdict</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L173-L182">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>asdict() -> Dict[str, int]
</code></pre>




<h3 id="total_t_count"><code>total_t_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L184-L205">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_t_count(
    ts_per_toffoli: int = 4,
    ts_per_cswap: int = 7,
    ts_per_and_bloq: int = 4,
    ts_per_rotation: int = 11
) -> int
</code></pre>

Get the total number of T Gates for the `GateCounts` object.

This simply multiplies each gate type by its cost in terms of T gates, which is configurable
via the arguments to this method.

The default value for `ts_per_rotation` assumes the rotation is approximated using
`Mixed fallback` protocol with error budget 1e-3.

<h3 id="total_t_and_ccz_count"><code>total_t_and_ccz_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L207-L210">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_t_and_ccz_count(
    ts_per_rotation: int = 11
) -> Dict[str, <a href="../../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>]
</code></pre>




<h3 id="total_beverland_count"><code>total_beverland_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L212-L235">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_beverland_count() -> Dict[str, <a href="../../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>]
</code></pre>

Counts used by Beverland. et. al. using notation from the reference.

 - $M_\mathrm{meas}$ is the number of measurements.
 - $M_R$ is the number of rotations.
 - $M_T$ is the number of T operations.
 - $3*M_mathrm{Tof}$ is the number of Toffoli operations.
 - $D_R$ is the number of layers containing at least one rotation. This can be smaller than
   the total number of non-Clifford layers since it excludes layers consisting only of T or
   Toffoli gates. Since we don't compile the 'layers' explicitly, we set this to be the
   number of rotations.

Reference




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class GateCounts.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class GateCounts.




