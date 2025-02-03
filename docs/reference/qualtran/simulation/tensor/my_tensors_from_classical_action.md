# my_tensors_from_classical_action


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/simulation/tensor/_tensor_from_classical.py#L106-L139">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the quimb tensors for the bloq derived from its `on_classical_vals` method.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>qualtran.simulation.tensor.my_tensors_from_classical_action(
    bloq: 'Bloq',
    incoming: dict[str, 'ConnectionT'],
    outgoing: dict[str, 'ConnectionT']
) -> list['qtn.Tensor']
</code></pre>



<!-- Placeholder for "Used in" -->

This function has the same signature as `bloq.my_tensors`, and can be used as a
replacement for it when the bloq has a known classical action.
For example:

```py
class ClassicalBloq(Bloq):
    ...

    def on_classical_vals(...):
        ...

    def my_tensors(self, incoming, outgoing):
        return my_tensors_from_classical_action(self, incoming, outgoing)
```