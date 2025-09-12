#  Copyright 2025 Google LLC
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

import qualtran.testing as qlt_testing
from qualtran.bloqs.hamiltonian_simulation.guided_hamiltonian import (
    _guided_hamiltonian_symb,
    _guided_phase_estimate_symb,
)


@pytest.mark.parametrize(
    "bloq_ex",
    [_guided_hamiltonian_symb, _guided_phase_estimate_symb],
    ids=lambda bloq_ex: bloq_ex.name,
)
def test_examples(bloq_autotester, bloq_ex):
    bloq_autotester(bloq_ex)


@pytest.mark.notebook
def test_notebook():
    qlt_testing.execute_notebook('guided_hamiltonian')


@pytest.mark.notebook
def test_tutorial():
    qlt_testing.execute_notebook('tutorial_guided_hamiltonian')
