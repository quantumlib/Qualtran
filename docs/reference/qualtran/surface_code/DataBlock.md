# DataBlock
`qualtran.surface_code.DataBlock`


<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L26-L112">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Methods for modeling the costs of the data block of a surface code compilation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`qualtran.surface_code.data_block.DataBlock`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

The number of algorithm qubits is reported by Qualtran as a logical cost of a bloq. The
surface code is a rate-1 code, so each bit of data needs at least one surface code tile. Due
to locality constraints imposed by the 2D surface code combined with the need to interact
qubits that aren’t necessarily local, additional tiles are needed to actually execute a program.

Each data block is responsible for reporting the number of tiles required to store a certain
number of algorithm qubits; as well as the number of time steps required to consume a magic
state. Different data blocks exist in the literature, and data block provides a different
space-time tradeoff.

The space occupied by the data block is to be contrasted with the space used for magic
state distillation.



<h2 class="add-link">Attributes</h2>

`data_d`<a id="data_d"></a>
: The code distance used to store the data in the data block.

`n_steps_to_consume_a_magic_state`<a id="n_steps_to_consume_a_magic_state"></a>
: The number of surface code steps to consume a magic state.
  
  We must teleport in "magic states" to do non-Clifford operations on our algorithmic
  data qubits. The layout of the data block can limit the number magic states consumed
  per unit time.
  
  One surface code step is `data_d` cycles of error correction.
  
  DataBlock imlpementation must override this method. This method is used by
  `self.n_cycles` to report the total number of cycles required.




## Methods

<h3 id="n_tiles"><code>n_tiles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L48-L68">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>n_tiles(
    n_algo_qubits: int
) -> int
</code></pre>

The number of surface code tiles used to store a given number of algorithm qubits.

 We define an “algorithm qubit” to be a qubit used in the routing of algorithm-relevant
 quantum data in a bloq. A physical qubit is a physical system that can encode one qubit,
 albeit noisily. Specific to the surface code, we define a “tile” to be the minimal area
 of physical qubits necessary to encode one logical qubit to a particular code distance.
 A tile can store an algorithm qubit, can be used for ancillary purposes like routing,
 or can be left idle. A tile is usually a square grid of $2d^2$ physical qubits.

 DataBlock implementations must override this method. This method is used by
 `self.n_phys_qubits` to report the total number of physical qubits.

Args

`n_algo_qubits`
: The number of algorithm qubits to compute the number of tiles for.




Returns




<h3 id="n_cycles"><code>n_cycles</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L85-L99">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_cycles(
    n_logical_gates: 'GateCounts', logical_error_model: 'LogicalErrorModel'
) -> int
</code></pre>

The number of surface code cycles to apply the number of gates to the data block.

Note that only the Litinski (2019) derived data blocks model a limit on the number of
magic states consumed per step. Other data blocks return "zero" for the number of cycles
due to the data block. When using those data block designs, it is assumed that the
number of cycles taken by the magic state factories is the limiting factor in the
computation.

<h3 id="n_physical_qubits"><code>n_physical_qubits</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L101-L104">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>n_physical_qubits(
    n_algo_qubits: int
) -> int
</code></pre>

The number of physical qubits used by the data block.


<h3 id="data_error"><code>data_error</code></h3>

<a target="_blank" class="external" href="https://github.com/quantumlib/Qualtran/blob/main/qualtran/surface_code/data_block.py#L106-L112">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>data_error(
    n_algo_qubits: int, n_cycles: int, logical_error_model: 'LogicalErrorModel'
) -> float
</code></pre>

The error associated with storing data on `n_algo_qubits` for `n_cycles`.




