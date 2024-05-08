# Module: qpic_diagram


Classes for drawing latex diagrams for bloqs with QPIC - https://github.com/qpic/qpic.



QPIC is not a dependency of Qualtran and must be manually installed by users via
`pip install qpic`.
## Classes

[`class QpicWireManager`](../../qualtran/drawing/qpic_diagram/QpicWireManager.md): Methods to manage allocation/deallocation of wires for QPIC diagrams.

[`class QpicCircuit`](../../qualtran/drawing/qpic_diagram/QpicCircuit.md): Builds data corresponding to the input specification of a QPIC diagram

## Functions

[`get_qpic_data(...)`](../../qualtran/drawing/qpic_diagram/get_qpic_data.md): Get the input data that can be used to draw a latex diagram for `bloq` using `qpic`.

[`qpic_diagram_for_bloq(...)`](../../qualtran/drawing/qpic_diagram/qpic_diagram_for_bloq.md): Generate latex diagram for `bloq` by invoking `qpic`. Assumes qpic is already installed.

[`qpic_input_to_diagram(...)`](../../qualtran/drawing/qpic_diagram/qpic_input_to_diagram.md): Invoke `qpic` script to generate output diagram of type qpic/tex/pdf/png.



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


