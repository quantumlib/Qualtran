# GateCounts
`qualtran.resource_counting.GateCounts`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L125-L277">
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L141-L153">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L155-L164">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L166-L167">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other
)
</code></pre>




<h3 id="asdict"><code>asdict</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L175-L184">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>asdict() -> Dict[str, int]
</code></pre>




<h3 id="total_t_count"><code>total_t_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L186-L207">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L209-L212">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_t_and_ccz_count(
    ts_per_rotation: int = 11
) -> Dict[str, SymbolicInt]
</code></pre>




<h3 id="total_toffoli_only"><code>total_toffoli_only</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L214-L220">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_toffoli_only() -> int
</code></pre>

The number of Toffoli-like gates, and raise an exception if there are Ts/rotations.


<h3 id="to_legacy_t_complexity"><code>to_legacy_t_complexity</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L222-L252">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_legacy_t_complexity(
    ts_per_toffoli: int = 4,
    ts_per_cswap: int = 7,
    ts_per_and_bloq: int = 4,
    cliffords_per_and_bloq: int = 9,
    cliffords_per_cswap: int = 10
) -> 'TComplexity'
</code></pre>

Return a legacy `TComplexity` object.

This coalesces all the gate types into t, rotations, and clifford fields. The conversion
factors can be tweaked using the arguments to this method.

The argument `cliffords_per_and_bloq` sets the base number of clifford gates to
add per `self.and_bloq`. To fully match the exact legacy `t_complexity` numbers, you
must enable `QECGatesCost(legacy_shims=True)`, which will enable a shim that directly
adds on clifford counts for the X-gates used to invert the And control lines.

<h3 id="total_beverland_count"><code>total_beverland_count</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/resource_counting/_bloq_counts.py#L254-L277">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>total_beverland_count() -> Dict[str, SymbolicInt]
</code></pre>

Counts used by Beverland et al. using notation from the reference.

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




