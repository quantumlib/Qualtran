#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from ._dense import bloq_to_dense, get_right_and_left_inds, quimb_to_dense
from ._flattening import bloq_has_custom_tensors, flatten_for_tensor_contraction
from ._quimb import cbloq_to_quimb, initialize_from_zero
from ._tensor_data_manipulation import (
    active_space_for_ctrl_spec,
    eye_tensor_for_signature,
    tensor_data_from_unitary_and_signature,
    tensor_out_inp_shape_from_signature,
    tensor_shape_from_signature,
)
