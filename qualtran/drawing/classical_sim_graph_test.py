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
import re

import numpy as np

from qualtran.bloqs.mcmt.and_bloq import MultiAnd
from qualtran.drawing import ClassicalSimGraphDrawer


def test_classical_sim_graph():
    gd = ClassicalSimGraphDrawer(
        bloq=MultiAnd((1, 1, 1, 1)).decompose_bloq(), vals=dict(ctrl=np.array([1, 1, 0, 1]))
    )

    # The main test is in the drawing notebook, so please spot check that.
    # Here: we make sure the edge labels are "0" and "1" for MultiAnd.
    dot_lines = gd.get_graph().to_string().splitlines()
    edge_lines = [line for line in dot_lines if '->' in line]
    for line in edge_lines:
        ma = re.search(r'label=(\w+)', line)
        assert ma is not None, line
        i = int(ma.group(1))
        assert i in [0, 1]
