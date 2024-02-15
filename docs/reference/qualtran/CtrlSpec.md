# CtrlSpec
`qualtran.CtrlSpec`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L32-L158">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification for how to control a bloq.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.CtrlSpec(
    qdtype: <a href="../qualtran/QDType.html"><code>qualtran.QDType</code></a> = <a href="../qualtran/QBit.html"><code>QBit()</code></a>,
    cvs: Union[int, NDArray[int], Iterable[int]] = 1
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class can be used by controlled bloqs to specify the condition under which the bloq
is active.

In the simplest form, a controlled gate is active when the control input is one qubit of data,
and it's in the |1> state. Otherwise, the gate is not performed. This corresponds to the
following two equivalent CtrlSpecs:

    CtrlSpec()
    CtrlSpec(qdtype=QBit(), cvs=1)

This class supports additional control specifications:
 1. 'negative' controls where the bloq is active if the input is |0>.
 2. integer-equality controls where a QInt input must match an integer control value.
 3. ndarrays of control values, where the bloq is active if **all** inputs are active.

For example: `CtrlSpec(cvs=[0, 1, 0, 1])` is active if the four input bits match the pattern.

A generalized control spec could support any number of "activation functions". The methods
`activation_function_dtypes` and `is_active` are defined for future extensibility.

<h2 class="add-link">Args</h2>

`qdtype`<a id="qdtype"></a>
: The quantum data type of the control input.

`cvs`<a id="cvs"></a>
: The control value(s). If more than one value is provided, they must all be
  compatible with `qdtype` and the bloq is implied to be active if **all** inputs
  are active.






<h2 class="add-link">Attributes</h2>

`cvs`<a id="cvs"></a>
: &nbsp;

`qdtype`<a id="qdtype"></a>
: &nbsp;

`shape`<a id="shape"></a>
: &nbsp;




## Methods

<h3 id="activation_function_dtypes"><code>activation_function_dtypes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L84-L98">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>activation_function_dtypes() -> Sequence[Tuple[QDType, Tuple[int, ...]]]
</code></pre>

The data types that serve as input to the 'activation function'.

The activation function takes in (quantum) inputs of these types and shapes and determines
whether the bloq should be active. This method is useful for setting up appropriate
control registers for a ControlledBloq.

This implementation returns one entry of type `self.qdtype` and shape `self.shape`.

Returns




<h3 id="is_active"><code>is_active</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L100-L126">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_active(
    *vals
) -> bool
</code></pre>

A classical implementation of the 'activation function'.

The activation function takes in (quantum) data and determines whether
the bloq should be active. This method captures the same behavior on specific classical
values representing computational basis states.

This implementation evaluates to `True` if all the values match `self.cvs`.

Args

`*vals`
: The classical values (that fit within the types given by
  `activation_function_dtypes`) on which we evaluate whether the spec is active.




Returns




<h3 id="wire_symbol"><code>wire_symbol</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L128-L137">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wire_symbol(
    i: int, soq: 'Soquet'
) -> 'WireSymbol'
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L139-L147">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: Any
) -> bool
</code></pre>

Return self==value.




