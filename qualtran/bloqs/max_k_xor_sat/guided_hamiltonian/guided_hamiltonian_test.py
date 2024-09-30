#  Copyright 2024 Google LLC
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
import pytest

from .guided_hamiltonian import _guided_hamiltonian, _guided_phase_estimate_symb


@pytest.mark.parametrize("bloq_ex", [_guided_hamiltonian, _guided_phase_estimate_symb])
def test_examples(bloq_autotester, bloq_ex):
    if bloq_autotester.check_name == 'serialize':
        pytest.skip()

    bloq_autotester(bloq_ex)


@pytest.mark.notebook
def test_notebook():
    from qualtran.testing import execute_notebook

    execute_notebook('guided_hamiltonian')
