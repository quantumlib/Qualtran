# QGFPoly
`qualtran.QGFPoly`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1060-L1159">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Univariate Polynomials with coefficients in a Galois Field GF($p^m$).

Inherits From: [`QDType`](../qualtran/QDType.md), [`QCDType`](../qualtran/QCDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.QGFPoly(
    degree, qgf
)
</code></pre>



<!-- Placeholder for "Used in" -->

This data type represents a degree-$n$ univariate polynomials
$f(x)=\sum_{i=0}^{n} a_i x^{i}$ where the coefficients $a_{i}$ of the polynomial
belong to a Galois Field $GF(p^{m})$.

The data type uses the [Galois library](https://mhostetter.github.io/galois/latest/) to
perform arithmetic over polynomials defined over Galois Fields using the
[galois.Poly](https://mhostetter.github.io/galois/latest/api/galois.Poly/).
References
    [Polynomials over finite fields](https://mhostetter.github.io/galois/latest/api/galois.Poly/)

    [Polynomial Arithmetic](https://mhostetter.github.io/galois/latest/basic-usage/poly-arithmetic/)



<h2 class="add-link">Attributes</h2>

`degree`<a id="degree"></a>
: The degree $n$ of the univariate polynomial $f(x)$ represented by this type.

`qgf`<a id="qgf"></a>
: An instance of `QGF` that represents the galois field $GF(p^m)$ over which the
  univariate polynomial $f(x)$ is defined.

`bitsize`<a id="bitsize"></a>
: &nbsp;

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1096-L1104">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="to_gf_coefficients"><code>to_gf_coefficients</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1110-L1114">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_gf_coefficients(
    f_x: galois.Poly
) -> galois.Array
</code></pre>

Returns a big-endian array of coefficients of the polynomial f(x).


<h3 id="from_gf_coefficients"><code>from_gf_coefficients</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1116-L1118">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_gf_coefficients(
    f_x: galois.Array
) -> galois.Poly
</code></pre>

Expects a big-endian array of coefficients that represent a polynomial f(x).


<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1120-L1124">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x
) -> List[int]
</code></pre>

Returns individual bits corresponding to binary representation of x


<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1126-L1129">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
)
</code></pre>

Combine individual bits to form x


<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1131-L1145">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val(
    val: Any, debug_str: str = &#x27;val&#x27;
)
</code></pre>

Raises an exception if `val` is not a valid classical value for this type.


Args

`val`
: A classical value that should be in the domain of this QDType.

`debug_str`
: Optional debugging information to use in exception messages.




<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1147-L1149">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this qdtype is parameterized with symbolic objects.


<h3 id="iteration_length_or_zero"><code>iteration_length_or_zero</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L1151-L1156">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iteration_length_or_zero() -> <a href="../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>
</code></pre>

Safe version of iteration length.

Returns the iteration_length if the type has it or else zero.

<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Check equality and either forward a NotImplemented or return the result negated.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class QGFPoly.




