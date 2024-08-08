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
from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import PrepareHubbard
from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import SelectHubbard
from qualtran.bloqs.qubitization.qubitization_walk_operator import QubitizationWalkOperator


def get_walk_operator_for_hubbard_model(
    x_dim: int, y_dim: int, t: int, u: int
) -> 'QubitizationWalkOperator':
    select = SelectHubbard(x_dim, y_dim)
    prepare = PrepareHubbard(x_dim, y_dim, t, u)

    return QubitizationWalkOperator(SelectBlockEncoding(select=select, prepare=prepare))
