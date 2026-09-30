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

from __future__ import annotations

import random
from typing import Any, Callable

import kickmix as km


def indentation_fix(code: str) -> str:
    saw_indentation = False
    for c in code:
        if c == '\n' or c == '\r':
            saw_indentation = False
            continue
        if c == ' ' or c == '\t':
            saw_indentation = True
            continue
        break
    if saw_indentation:
        code = 'if True:\n' + code
    code = """
_global_phase = False
def Z(b):
    global _global_phase
    _global_phase ^= b
def neg():
    Z(True)
""" + code
    return code


def assert_fuzz_testing_acts_like(
    circuit: km.Circuit,
    code: str,
    *,
    shots: int,
    context: dict[str, Any] | None = None,
    input_sampler: Callable[[], dict[str, int]] | None = None,
    ignore_phase: bool = False,
):
    __tracebackhide__ = True
    if context is None:
        context = {}
    register_data = circuit.register_data
    for k, v in register_data.items():
        if not isinstance(k, str):
            raise AssertionError(f"Unnamed register {k!r}")

    unused_qubits = set(km.q(k) for k in range(circuit.num_qubits))
    for v in register_data.values():
        for q in v:
            unused_qubits.discard(q)
    unused_array = km.array(sorted(unused_qubits))

    sim = km.Simulator(batch_size=min(shots, 256))
    sim.use_same_registers_as(circuit)

    all_inputs = []
    all_expected = []
    all_actuals = []
    shots_remaining = shots
    while shots_remaining > 0:
        shot_range = range(min(shots_remaining, 256))
        shots_remaining -= len(shot_range)
        sim.clear_for_shot()

        for shot in shot_range:
            inputs = {}
            if input_sampler is None:
                for k, v in register_data.items():
                    inputs[k] = random.randrange(1 << len(v))
                    sim.write_within_shot(k, shot, inputs[k])
            else:
                sample = input_sampler()
                assert sample.keys() == register_data.keys()
                for k, v in register_data.items():
                    inputs[k] = sample[k]
                    sim.write_within_shot(k, shot, sample[k])
            ctx = {**inputs, **context}
            out = {}
            exec(indentation_fix(code), out, ctx)
            expected = {}
            for k, v in register_data.items():
                expected[k] = ctx[k]
            expected['@phase'] = out['_global_phase']
            all_inputs.append(inputs)
            all_expected.append(expected)

        sim.do(circuit)
        for shot in shot_range:
            actual = {}
            for k, v in register_data.items():
                actual[k] = sim.read_within_shot(k, shot, out=int)
            actual['@unused'] = sim.read_within_shot(unused_array, shot, out=int)
            actual['@phase'] = sim.read_shot_phase(shot)
            all_actuals.append(actual)

        for shot in shot_range:
            actual = all_actuals[shot]
            expected = all_expected[shot]
            inp = all_inputs[shot]
            failed = False
            for k, v in register_data.items():
                failed |= actual[k] != expected[k]
            failed |= actual['@unused'] != 0
            if not ignore_phase:
                failed |= actual['@phase'] != expected['@phase']
            if failed:
                lines = ["Fuzz testing failed"]
                for k, v in register_data.items():
                    if actual[k] != expected[k]:
                        line = f"    FAIL:register {k!r} = {inp[k]} -> {actual[k]} != {expected[k]}"
                    else:
                        line = f"    register {k!r} = {inp[k]} -> {actual[k]}"
                    lines.append(line)
                if actual["@phase"] != expected['@phase']:
                    lines.append(f"""    FAIL:@phase={actual["@phase"]} != {expected["@phase"]})""")
                else:
                    lines.append("    @phase=" + str(actual["@phase"]))
                if actual["@unused"]:
                    lines.append("    FAIL:@unused=" + str(actual["@unused"]))
                else:
                    lines.append("    @unused=" + str(actual["@unused"]))
                for k, v in context.items():
                    lines.append(f"    var[{k!r}] = {v!r}")
                raise AssertionError('\n'.join(lines))
