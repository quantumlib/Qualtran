# QMontgomeryUInt
`qualtran.QMontgomeryUInt`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L793-L900">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Montgomery form of an unsigned integer of a given width bitsize which wraps around upon overflow.

Inherits From: [`QDType`](../qualtran/QDType.md), [`QCDType`](../qualtran/QCDType.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.QMontgomeryUInt(
    bitsize, modulus=attr_dict[&#x27;modulus&#x27;].default
)
</code></pre>



<!-- Placeholder for "Used in" -->

Similar to unsigned integer types in C. Any intended wrap around effect is
expected to be handled by the developer. Any QMontgomeryUInt can be treated as a QUInt, but not
every QUInt can be treated as a QMontgomeryUInt. Montgomery form is used in order to compute
fast modular multiplication.

In order to convert an unsigned integer from a finite field x % p into Montgomery form you
first must choose a value r > p where gcd(r, p) = 1. Typically, this value is a power of 2.

Conversion to Montgomery form:
    [x] = (x * r) % p

Conversion from Montgomery form to normal form:
    x = REDC([x])

Pseudocode for REDC(u) can be found in the resource below.

<h2 class="add-link">References</h2>






<h2 class="add-link">Attributes</h2>

`bitsize`<a id="bitsize"></a>
: The number of qubits used to represent the integer.

`modulus`<a id="modulus"></a>
: &nbsp;

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="is_symbolic"><code>is_symbolic</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L832-L833">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_symbolic() -> bool
</code></pre>

Returns True if this dtype is parameterized with symbolic objects.


<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L835-L838">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L840-L842">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits(
    x: int
) -> List[int]
</code></pre>

Yields individual bits corresponding to binary representation of x


<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L844-L845">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_bits(
    bits: Sequence[int]
) -> int
</code></pre>

Combine individual bits to form x


<h3 id="assert_valid_classical_val"><code>assert_valid_classical_val</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L847-L853">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val(
    val: int, debug_str: str = &#x27;val&#x27;
)
</code></pre>

Raises an exception if `val` is not a valid classical value for this type.


Args

`val`
: A classical value that should be in the domain of this QDType.

`debug_str`
: Optional debugging information to use in exception messages.




<h3 id="assert_valid_classical_val_array"><code>assert_valid_classical_val_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L855-L861">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>assert_valid_classical_val_array(
    val_array: NDArray[np.integer], debug_str: str = &#x27;val&#x27;
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




<h3 id="montgomery_inverse"><code>montgomery_inverse</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L863-L872">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>montgomery_inverse(
    xm: int
) -> int
</code></pre>

Returns the modular inverse of an integer in montgomery form.


Args

`xm`
: An integer in montgomery form.




<h3 id="montgomery_product"><code>montgomery_product</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L874-L882">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>montgomery_product(
    xm: int, ym: int
) -> int
</code></pre>

Returns the modular product of two integers in montgomery form.


Args

`xm`
: The first montgomery form integer for the product.

`ym`
: The second montgomery form integer for the product.




<h3 id="montgomery_to_uint"><code>montgomery_to_uint</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L884-L891">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>montgomery_to_uint(
    xm: int
) -> int
</code></pre>

Converts an integer in montgomery form to a normal form integer.


Args

`xm`
: An integer in montgomery form.




<h3 id="uint_to_montgomery"><code>uint_to_montgomery</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L893-L900">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>uint_to_montgomery(
    x: int
) -> int
</code></pre>

Converts an integer into montgomery form.


Args

`x`
: An integer.




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

Method generated by attrs for class QMontgomeryUInt.




