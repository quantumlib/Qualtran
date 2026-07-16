#  Copyright 2026 Google LLC
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

from rsqualtran import QLTFastsim

import qualtran.dtype as qdt
from qualtran.bloqs.arithmetic import Negate


def main():
    bloq = Negate(qdt.QInt(8))
    simulator = QLTFastsim.from_bloq(bloq)
    print(simulator)
    for x in range(2**7):
        (result_x,) = simulator.call_classically(x=x)  # pylint: disable=unbalanced-tuple-unpacking
        print(f"x={x} -> result_x={result_x}")


if __name__ == "__main__":
    main()
