# CtrlSpec
`qualtran.CtrlSpec`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L59-L232">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A specification for how to control a bloq.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.CtrlSpec(
    qdtypes=attr_dict[&#x27;qdtypes&#x27;].default,
    cvs=attr_dict[&#x27;cvs&#x27;].default
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class can be used by controlled bloqs to specify the condition under which the bloq
is active.

In the simplest form, a controlled gate is active when the control input is one qubit of data,
and it's in the |1> state. Otherwise, the gate is not performed. This corresponds to the
following two equivalent CtrlSpecs:

    CtrlSpec()
    CtrlSpec(qdtypes=QBit(), cvs=1)

This class supports additional control specifications:
 1. 'negative' controls where the bloq is active if the input is |0>.
 2. integer-equality controls where a QInt input must match an integer control value.
 3. ndarrays of control values, where the bloq is active if **all** inputs are active.
 4. Multiple control registers, control values for each of which can be specified
    using 1-3 above.

#### For example:


1. `CtrlSpec(qdtypes=QUInt(4), cvs=0b0110)`:
        Ctrl for a single register, of type `QUInt(4)` and shape `()`, is active when the
        soquet of the input register takes value 6.
2. `CtrlSpec(cvs=[0, 1, 1, 0])`:
        Ctrl for a single register, of type `QBit()` and shape `(4,)`, is active when soquets
        of input register take values `[0, 1, 1, 0]`.
3. `CtrlSpec(qdtypes=[QBit(), QBit()], cvs=[[0, 1], [1, 0]]).is_active([0, 1], [1, 0])`:
        Ctrl for 2 registers, each of type `QBit()` and shape `(2,)`, is active when the
        soquet for each register takes values `[0, 1]` and  `[1, 0]` respectively.

CtrlSpec uses logical AND among all control register clauses. If you need a different boolean
function, open a GitHub issue.

<h2 class="add-link">Args</h2>

`qdtypes`<a id="qdtypes"></a>
: A tuple of quantum data types, one per ctrl register.

`cvs`<a id="cvs"></a>
: A tuple of control value(s), one per ctrl register. For each element in the tuple,
  if more than one ctrl value is provided, they must all be compatible with `qdtype`
  and the bloq is implied to be active if **all** inputs are active (i.e. the "shape"
  of the ctrl register is implied to be `cv.shape`).






<h2 class="add-link">Attributes</h2>

`cvs`<a id="cvs"></a>
: &nbsp;

`num_ctrl_reg`<a id="num_ctrl_reg"></a>
: &nbsp;

`qdtypes`<a id="qdtypes"></a>
: &nbsp;

`shapes`<a id="shapes"></a>
: &nbsp;




## Methods

<h3 id="activation_function_dtypes"><code>activation_function_dtypes</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L119-L129">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>activation_function_dtypes() -> Sequence[Tuple[QDType, Tuple[int, ...]]]
</code></pre>

The data types that serve as input to the 'activation function'.

The activation function takes in (quantum) inputs of these types and shapes and determines
whether the bloq should be active. This method is useful for setting up appropriate
control registers for a ControlledBloq.

Returns




<h3 id="is_active"><code>is_active</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L131-L159">View source</a>

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

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L161-L170">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wire_symbol(
    i: int, soq: 'Soquet'
) -> 'WireSymbol'
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L176-L184">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: Any
) -> bool
</code></pre>

Return self==value.


<h3 id="to_cirq_cv"><code>to_cirq_cv</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L189-L195">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_cirq_cv() -> cirq.SumOfProducts
</code></pre>

Convert CtrlSpec to cirq.SumOfProducts representation of control values.


<h3 id="from_cirq_cv"><code>from_cirq_cv</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/_infra/controlled.py#L197-L232">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_cirq_cv(
    cirq_cv: cirq.ops.AbstractControlValues,
    *,
    qdtypes: Optional[Sequence[QDType]] = None,
    shapes: Optional[Sequence[Tuple[int, ...]]] = None
) -> 'CtrlSpec'
</code></pre>

Construct a CtrlSpec from cirq.SumOfProducts representation of control values.




