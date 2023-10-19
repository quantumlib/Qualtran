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

import numpy as np

from qualtran.bloqs.chemistry.first_quantization import PrepareKineticFirstQuantization
from qualtran.testing import execute_notebook


def _make_prepare_kinetic():
    from qualtran.bloqs.chemistry.first_quantization import PrepareKineticFirstQuantization

    num_pw = (2 * 10 + 1) ** 3
    eta = 10
    return PrepareKineticFirstQuantization(num_pw, eta)


def test_notebook():
    execute_notebook('first_quantization')
