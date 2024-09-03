# NoisyPauliRotation
`qualtran.surface_code.t_factory_utils.NoisyPauliRotation`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/t_factory_utils.py#L28-L60">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A channel that applies a pi/8 pauli rotation with possible overshooting to 5pi/8, -pi/8, and 3pi/8.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.surface_code.t_factory_utils.NoisyPauliRotation(
    pauli_string: str, p1: float, p2: float, p3: float
)
</code></pre>



<!-- Placeholder for "Used in" -->

The channel is defined as

$$
    \sum_{k \in \{1, 5, -1, 3\}} p_k e^{-i \frac{\pi}{8} k P} \rho e^{i \frac{\pi}{8} k P}
$$

<h2 class="add-link">Args</h2>

`pauli_string`<a id="pauli_string"></a>
: The pauli string to apply the rotation to.

`p1`<a id="p1"></a>
: The probability of applying the rotation by 5pi/8.

`p2`<a id="p2"></a>
: The probability of applying the rotation by -pi/8.

`p3`<a id="p3"></a>
: The probability of applying the rotation by 3pi/8.




## Methods

<h3 id="num_qubits"><code>num_qubits</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.




