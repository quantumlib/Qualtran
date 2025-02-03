# QGF
`qualtran.QGF`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L866-L989">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Galois Field type to represent elements of a finite field.

Inherits From: [`QDType`](../qualtran/QDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.QGF(
    characteristic, degree
)
</code></pre>



<!-- Placeholder for "Used in" -->

A Finite Field or Galois Field is a field that contains finite number of elements. The order
of a finite field is the number of elements in the field, which is either a prime number or
a prime power. For every prime number $p$ and every positive integer $m$ there are fields of
order $p^m$, all of which are isomorphic. When m=1, the finite field of order p can be
constructed via integers modulo p.

Elements of a Galois Field $GF(p^m)$ may be conveniently viewed as polynomials
$a_{0} + a_{1}x + ... + a_{m−1}x_{m−1}$, where $a_0, a_1, ..., a_{m−1} \in F(p)$.
$GF(p^m)$ addition is defined as the component-wise (polynomial) addition over F(p) and
multiplication is defined as polynomial multiplication modulo an irreducible polynomial of
degree $m$. The selection of the specific irreducible polynomial affects the representation
of the given field, but all fields of a fixed size are isomorphic.

The data type uses the [Galois library](https://mhostetter.github.io/galois/latest/) to
perform arithmetic over Galois Fields. By default, the Conway polynomial $C_{p, m}$ is used
as the irreducible polynomial.
References
    [Finite Field](https://en.wikipedia.org/wiki/Finite_field)

    [Intro to Prime Fields](https://mhostetter.github.io/galois/latest/tutorials/intro-to-prime-fields/)

    [Intro to Extension Fields](https://mhostetter.github.io/galois/latest/tutorials/intro-to-extension-fields/)



<h2 class="add-link">Attributes</h2>

`characteristic`<a id="characteristic"></a>
: The characteristic $p$ of the field $GF(p^m)$.
  The characteristic must be prime.

`degree`<a id="degree"></a>
: The degree $m$ of the field $GF(p^{m})$. The degree must be a positive integer.

`bitsize`<a id="bitsize"></a>
: &nbsp;

`gf_type`<a id="gf_type"></a>
: &nbsp;

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.

`order`<a id="order"></a>
: &nbsp;




## Methods

<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L917-L920">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L932-L935">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x
) -> List[int]
</code></pre>

Yields individual bits corresponding to binary representation of x


<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L937-L939">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
)
</code></pre>

Combine individual bits to form x


<h3 id="from_bits_array"><code>from_bits_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L941-L948">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits_array(
    bits_array: NDArray[np.uint8]
)
</code></pre>

Combine individual bits to form classical values.

Often, converting an array can be performed faster than converting each element individually.
This operation accepts any NDArray of bits such that the last dimension equals `self.bitsize`,
and the output array satisfies `output_shape = input_shape[:-1]`.

<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L950-L958">View source</a>

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




<h3 id="assert_valid_classical_val_array"><code>assert_valid_classical_val_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L960-L975">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val_array(
    val_array: NDArray[Any], debug_str: str = &#x27;val&#x27;
)
</code></pre>

Raises an exception if `val_array` is not a valid array of classical values for this type.

Often, validation on an array can be performed faster than validating each element
individually.

Args

`val_array`
: A numpy array of classical values. Each value should be in the domain
  of this QDType.

`debug_str`
: Optional debugging information to use in exception messages.




<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L977-L979">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this qdtype is parameterized with symbolic objects.


<h3 id="iteration_length_or_zero"><code>iteration_length_or_zero</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L981-L986">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iteration_length_or_zero() -> <a href="../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>
</code></pre>

Safe version of iteration length.

Returns the iteration_length if the type has it or else zero.

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Method generated by attrs for class QGF.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Method generated by attrs for class QGF.




