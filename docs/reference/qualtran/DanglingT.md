# DanglingT
`qualtran.DanglingT`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/quantum_graph.py#L52-L72">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The type of the singleton objects `LeftDangle` and `RightDangle`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.DanglingT(
    x
)
</code></pre>



<!-- Placeholder for "Used in" -->

These objects are placeholders for the `binst` field of a `Soquet` that represents
an "external wire". We can consider `Soquets` of this type to represent input or
output data of a `CompositeBloq`.

## Methods

<h3 id="bloq_is"><code>bloq_is</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/cirq-qubitization/blob/main/qualtran/_infra/quantum_graph.py#L66-L72">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>bloq_is(
    t
) -> bool
</code></pre>

DanglingT.bloq_is(...) is always False.

This is to support convenient isinstance checking on binst.bloq where
binst may be a `DanglingT`.



