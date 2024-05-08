# LineManager
`qualtran.drawing.LineManager`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L87-L192">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Methods to manage allocation and de-allocation of lines representing a register of qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.drawing.musical_score.LineManager`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.drawing.LineManager(
    max_n_lines: int = 100
)
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="new_y"><code>new_y</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L96-L98">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>new_y(
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>,
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>,
    idx=None
)
</code></pre>

Allocate a new y position (i.e. a new qubit or register).


<h3 id="reserve_n"><code>reserve_n</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L100-L110">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reserve_n(
    n: int, until
)
</code></pre>

Reserve `n` lines until further notice.

To have fine-grained control over the vertical layout of HLines, consider
overriding `maybe_reserve` which can call this method to reserve lines
depending on the musical score context.

<h3 id="unreserve"><code>unreserve</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L112-L121">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unreserve(
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>,
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>
)
</code></pre>

Go through our reservations and rescind them depending on the `until` predicate.


<h3 id="maybe_reserve"><code>maybe_reserve</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L123-L137">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>maybe_reserve(
    binst: Union[<a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>, <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>],
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>,
    idx: Tuple[int, ...]
)
</code></pre>

Override this method to provide custom control over line allocation.

After a new y position is allocated and after a y position is freed, this method
is called  with the current `binst, reg, idx`. You can inspect these elements to
determine whether you want to continue allocating lines first-come-first-serve by
returning without doing anything;
or you can call `self.reserve_n(n, until)` to keep the next `n` lines unavailable
until the `until` callback predicate evaluates to True.

Whenever a new register is encountered, we first go through existing reservations
and call the `until` predicate on `binst, reg`.

<h3 id="new"><code>new</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L139-L160">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>new(
    binst: <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>,
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>,
    seq_x: int,
    topo_gen: int
) -> Union[RegPosition, NDArray[RegPosition]]
</code></pre>

Allocate a position or positions for `reg`.

`binst` and `reg` can optionally modify the allocation strategy.
`seq_x` and `topo_gen` are passed through.

<h3 id="finish_hline"><code>finish_hline</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L162-L166">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>finish_hline(
    y: int, seq_x_end: int
)
</code></pre>

Update `self.hlines` once we know where an HLine ends.


<h3 id="free"><code>free</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/drawing/musical_score.py#L168-L192">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>free(
    binst: Union[<a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>, <a href="../../qualtran/BloqInstance.html"><code>qualtran.BloqInstance</code></a>],
    reg: <a href="../../qualtran/Register.html"><code>qualtran.Register</code></a>,
    arr: Union[RegPosition, NDArray[RegPosition]]
)
</code></pre>

De-allocate a position or positions for `reg`.

This will free the position for future allocation. This will find the in-progress
HLine associate with `reg` and update it to indicate the end point.



