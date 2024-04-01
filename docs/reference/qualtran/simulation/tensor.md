# Module: tensor






## Functions

[`active_space_for_ctrl_spec(...)`](../../qualtran/simulation/tensor/active_space_for_ctrl_spec.md): Returns the "active" subspace corresponding to `signature` and `ctrl_spec`.

[`bloq_has_custom_tensors(...)`](../../qualtran/simulation/tensor/bloq_has_custom_tensors.md): Whether this bloq declares custom tensors by overriding `.add_my_tensors(...)`.

[`bloq_to_dense(...)`](../../qualtran/simulation/tensor/bloq_to_dense.md): Return a contracted, dense ndarray representing the composite bloq.

[`cbloq_as_contracted_tensor(...)`](../../qualtran/simulation/tensor/cbloq_as_contracted_tensor.md): `add_my_tensors` helper for contracting `cbloq` and adding it as a dense tensor.

[`cbloq_to_quimb(...)`](../../qualtran/simulation/tensor/cbloq_to_quimb.md): Convert a composite bloq into a Quimb tensor network.

[`eye_tensor_for_signature(...)`](../../qualtran/simulation/tensor/eye_tensor_for_signature.md): Returns an identity tensor with shape `tensor_shape_from_signature(signature)`

[`flatten_for_tensor_contraction(...)`](../../qualtran/simulation/tensor/flatten_for_tensor_contraction.md): Flatten a (composite) bloq as much as possible to enable efficient tensor contraction.

[`get_right_and_left_inds(...)`](../../qualtran/simulation/tensor/get_right_and_left_inds.md): Return right and left tensor indices.

[`tensor_data_from_unitary_and_signature(...)`](../../qualtran/simulation/tensor/tensor_data_from_unitary_and_signature.md): Returns tensor data respecting `signature` corresponding to `unitary`

[`tensor_out_inp_shape_from_signature(...)`](../../qualtran/simulation/tensor/tensor_out_inp_shape_from_signature.md): Returns a tuple for tensor data corresponding to signature.

[`tensor_shape_from_signature(...)`](../../qualtran/simulation/tensor/tensor_shape_from_signature.md): Returns a tuple for tensor data corresponding to signature.

