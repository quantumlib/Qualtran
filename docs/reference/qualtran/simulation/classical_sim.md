# Module: classical_sim


Functionality for the <a href="../../qualtran/Bloq.html#call_classically"><code>Bloq.call_classically(...)</code></a> protocol.



## Classes

[`class ClassicalValDistribution`](../../qualtran/simulation/classical_sim/ClassicalValDistribution.md): This class represents a distribution of classical values.

[`class MeasurementPhase`](../../qualtran/simulation/classical_sim/MeasurementPhase.md): Sentinel value for phases based on measurement outcomes:

[`class ClassicalSimState`](../../qualtran/simulation/classical_sim/ClassicalSimState.md): A mutable class for classically simulating composite bloqs.

[`class PhasedClassicalSimState`](../../qualtran/simulation/classical_sim/PhasedClassicalSimState.md): A mutable class for classically simulating composite bloqs with phase tracking.

## Functions

[`add_ints(...)`](../../qualtran/simulation/classical_sim/add_ints.md): Performs addition modulo $2^\mathrm{num\_bits}$ of (un)signed in a reversible way.

[`call_cbloq_classically(...)`](../../qualtran/simulation/classical_sim/call_cbloq_classically.md): Propagate `on_classical_vals` calls through a composite bloq's contents.

[`do_phased_classical_simulation(...)`](../../qualtran/simulation/classical_sim/do_phased_classical_simulation.md): Do a phased classical simulation of the bloq.

[`format_classical_truth_table(...)`](../../qualtran/simulation/classical_sim/format_classical_truth_table.md): Get a formatted tabular representation of the classical truth table.

[`get_classical_truth_table(...)`](../../qualtran/simulation/classical_sim/get_classical_truth_table.md): Get a 'truth table' for a classical-reversible bloq.

## Type Aliases

[`ClassicalValRetT`](../../qualtran/simulation/classical_sim/ClassicalValRetT.md)

[`ClassicalValT`](../../qualtran/simulation/classical_sim/ClassicalValT.md)

[`NDArray`](../../qualtran/testing/NDArray.md)



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


