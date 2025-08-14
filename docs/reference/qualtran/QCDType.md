# QCDType
`qualtran.QCDType`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L31-L117">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The abstract interface for quantum/classical quantum computing data types.

<!-- Placeholder for "Used in" -->




<h2 class="add-link">Attributes</h2>

`num_bits`<a id="num_bits"></a>
: &nbsp;

`num_cbits`<a id="num_cbits"></a>
: Number of classical bits required to represent a single instance of this data type.

`num_qubits`<a id="num_qubits"></a>
: Number of qubits required to represent a single instance of this data type.




## Methods

<h3 id="get_classical_domain"><code>get_classical_domain</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L48-L51">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get_classical_domain() -> Iterable[Any]
</code></pre>

Yields all possible classical (computational basis state) values representable by this type.


<h3 id="to_bits"><code>to_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L53-L55">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>to_bits(
    x
) -> List[int]
</code></pre>

Yields individual bits corresponding to binary representation of x


<h3 id="to_bits_array"><code>to_bits_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L57-L66">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_bits_array(
    x_array: NDArray[Any]
) -> NDArray[np.uint8]
</code></pre>

Yields an NDArray of bits corresponding to binary representations of the input elements.

Often, converting an array can be performed faster than converting each element individually.
This operation accepts any NDArray of values, and the output array satisfies
`output_shape = input_shape + (self.bitsize,)`.

<h3 id="from_bits"><code>from_bits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L68-L70">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>from_bits(
    bits: Sequence[int]
)
</code></pre>

Combine individual bits to form x


<h3 id="from_bits_array"><code>from_bits_array</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L72-L79">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L81-L88">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L90-L103">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L105-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>is_symbolic() -> bool
</code></pre>

Returns True if this dtype is parameterized with symbolic objects.


<h3 id="iteration_length_or_zero"><code>iteration_length_or_zero</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/data_types.py#L109-L114">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iteration_length_or_zero() -> <a href="../qualtran/symbolics/SymbolicInt.html"><code>qualtran.symbolics.SymbolicInt</code></a>
</code></pre>

Safe version of iteration length.

Returns the iteration_length if the type has it or else zero.



