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

"""Bloqs to load classical data in a quantum register"""

from qualtran.bloqs.data_loading.qroam_clean import QROAMClean, QROAMCleanAdjoint
from qualtran.bloqs.data_loading.qrom import QROM
from qualtran.bloqs.data_loading.select_swap_qrom import SelectSwapQROM
