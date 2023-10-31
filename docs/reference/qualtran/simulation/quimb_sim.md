# Module: quimb_sim


Functionality for the <a href="../../qualtran/Bloq.html#tensor_contract"><code>Bloq.tensor_contract()</code></a> protocol.



## Functions

[`bloq_has_custom_tensors(...)`](../../qualtran/simulation/quimb_sim/bloq_has_custom_tensors.md): Whether this bloq declares custom tensors by overriding `.add_my_tensors(...)`.

[`cbloq_to_quimb(...)`](../../qualtran/simulation/quimb_sim/cbloq_to_quimb.md): Convert a composite bloq into a Quimb tensor network.

[`flatten_for_tensor_contraction(...)`](../../qualtran/simulation/quimb_sim/flatten_for_tensor_contraction.md): Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.

[`get_right_and_left_inds(...)`](../../qualtran/simulation/quimb_sim/get_right_and_left_inds.md): Return right and left indices.



<h2 class="add-link">Other Members</h2>

LeftDangle<a id="LeftDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>

RightDangle<a id="RightDangle"></a>
: Instance of <a href="../../qualtran/DanglingT.html"><code>qualtran.DanglingT</code></a>


