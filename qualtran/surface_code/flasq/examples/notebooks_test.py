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
"""Tests for FLASQ example notebooks."""
import os
import tempfile
import pytest
from qualtran.testing import execute_notebook

def _execute_notebook_in_temp_dir(name: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            execute_notebook(name)
        finally:
            os.chdir(old_cwd)

@pytest.mark.notebook
def test_ising_notebook():
    os.environ['FLASQ_FAST_MODE_OVERRIDE'] = 'True'
    os.environ['MPLBACKEND'] = 'Agg'
    _execute_notebook_in_temp_dir('ising_notebook')

@pytest.mark.notebook
def test_hwp_notebook():
    os.environ['FLASQ_FAST_MODE_OVERRIDE'] = 'True'
    os.environ['MPLBACKEND'] = 'Agg'
    _execute_notebook_in_temp_dir('hwp_notebook')

@pytest.mark.notebook
def test_gf2_multiplier_notebook():
    os.environ['FLASQ_FAST_MODE_OVERRIDE'] = 'True'
    os.environ['MPLBACKEND'] = 'Agg'
    _execute_notebook_in_temp_dir('gf2_multiplier_example_notebook')
