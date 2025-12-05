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

import cirq
import numpy as np
import pytest

import qualtran.rotation_synthesis._math_config as mc
import qualtran.rotation_synthesis.channels as ch
import qualtran.rotation_synthesis.matrix._clifford_t_repr as ctr
import qualtran.rotation_synthesis.matrix._su2_ct as _su2_ct


def _make_gates(n_seqs: int, n_gates: int, seed: int):
    gates = tuple(_su2_ct.GATE_MAP.keys())
    rng = np.random.default_rng(seed)
    for _ in range(n_seqs):
        yield [gates[i] for i in rng.choice(len(gates), n_gates)]


@pytest.mark.parametrize("gates", _make_gates(10, 4, 0))
def test_unitary_from_sequence(gates):
    u = ch.UnitaryChannel.from_sequence(gates)
    assert u.to_matrix() == _su2_ct.SU2CliffordT.from_sequence(gates)


@pytest.mark.parametrize(
    ["gates", "theta", "distance"],
    [
        [["I", "Z", "I", "Tz"], 4.179974975284761, 0.2785216431390912],
        [["I", "S", "Tz"], 1.788395114150811, 0.34841379843764864],
        [["I", "S", "Tz"], 2.147407592736323, 0.36575434272229357],
        [["I", "Z", "S", "Tz"], 0.5569130698464282, 0.3269538876258962],
        [["I", "S"], 5.528036816451128, 0.06049011917942428],
    ],
)
def test_diamond_distance_for_unitary(gates, theta, distance):
    c = ch.UnitaryChannel.from_sequence(gates)
    np.testing.assert_allclose(c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance)
    u = np.zeros((2, 2), complex)
    u[1, 1] = np.exp(-1j * theta)
    u[0, 0] = u[1, 1].conjugate()
    np.testing.assert_allclose(c.diamond_norm_distance_to_unitary(u, mc.NumpyConfig), distance)


@pytest.mark.parametrize(
    ["proj", "correction", "theta", "distance"],
    [
        [
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            ["I", "Z", "S", "H", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx", "Ty", "Tz", "Tx"],
            4.058393950738398,
            0.12265553387764809,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 0.7154420837719515, 0.13979806891313962],
        [
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            ["I", "Z", "S", "H", "Tz", "Tx", "Ty", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx"],
            4.041343072659278,
            0.15656986343894638,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 0.8609971423751847, 0.1510539778688528],
        [
            ["I", "H", "S", "Ty", "Tx", "Tz", "Ty", "Tx"],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            2.910895329185765,
            0.18545310121849135,
        ],
        [
            ["I", "Y", "H", "S", "H", "Ty", "Tx", "Ty", "Tz", "Tx"],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            0.5337113492446506,
            0.22683760663945227,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 4.299933159455965, 0.7287141780613403],
        [
            ["I", "H", "S", "Tx", "Ty", "Tz", "Ty", "Tz", "Ty", "Tx"],
            ["I", "S", "H", "Ty", "Tz", "Ty", "Tx", "Ty", "Tz", "Ty", "Tx", "Ty"],
            2.4629437050180374,
            0.17169223895328464,
        ],
        [["I", "Z", "S"], ["I", "Z", "S"], 3.9026689436493815, 0.04863895092095243],
    ],
)
def test_diamond_distance_for_fallback(proj, correction, theta, distance):
    c = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj), ch.UnitaryChannel.from_sequence(correction)
    )
    np.testing.assert_allclose(c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance)


@pytest.mark.parametrize(
    ["seq1", "seq2", "theta", "distance"],
    [
        [
            ["I", "I", "Ty", "Tx", "Ty", "Tz", "Ty", "Tz", "Tx", "Tz", "Ty", "Tx", "Tz", "Ty"],
            ["I", "Y", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty"],
            4.101555349367053,
            0.007986955439257544,
        ],
        [
            ["I", "Z", "H", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx"],
            ["I", "Z", "H", "Ty", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz", "Tx", "Ty"],
            3.959743679319142,
            0.0023654253421858873,
        ],
        [
            ["I", "H", "Tx", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx"],
            ["I", "H", "Ty", "Tx", "Ty", "Tz", "Tx", "Tz", "Tx", "Ty", "Tz", "Ty", "Tz"],
            5.18570635613891,
            0.004288062424341757,
        ],
        [
            ["I", "S", "H", "Tx", "Ty", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz", "Tx", "Ty"],
            ["I", "S", "H", "Tz", "Tx", "Ty", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx", "Tz"],
            4.442270210803678,
            0.002937255469622333,
        ],
        [
            ["I", "X", "I", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz"],
            ["I", "S", "H", "Tz", "Ty", "Tz", "Ty", "Tx", "Ty", "Tx"],
            2.160269652544976,
            0.006305821840210275,
        ],
        [
            ["I", "Y", "I", "Tz", "Tx", "Tz", "Ty", "Tx", "Ty", "Tz", "Ty"],
            [
                "I",
                "Z",
                "H",
                "S",
                "H",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Tx",
                "Tz",
                "Ty",
                "Tx",
                "Tz",
                "Ty",
                "Tz",
            ],
            4.169354523270907,
            0.006945248214462576,
        ],
        [
            ["I", "H", "Tx", "Tz", "Tx", "Ty", "Tx", "Tz", "Tx", "Tz", "Ty", "Tz", "Tx"],
            ["I", "X", "H", "S", "Tx", "Tz", "Tx", "Tz", "Ty", "Tx", "Ty", "Tx", "Tz", "Ty"],
            1.9218237240459548,
            0.005494560413584842,
        ],
        [
            ["I", "X", "H", "S", "Tz", "Ty", "Tz", "Ty", "Tx", "Tz", "Tx", "Tz", "Ty", "Tx", "Ty"],
            ["I", "H", "S", "H", "Tz", "Ty", "Tx", "Tz", "Ty", "Tx", "Ty", "Tx", "Ty", "Tx"],
            0.0203606083765623,
            0.0003712210316894235,
        ],
        [
            ["I", "H", "Tx", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx"],
            ["I", "H", "Tx", "Ty", "Tz", "Tx", "Ty", "Tx", "Tz", "Ty", "Tz", "Tx", "Ty"],
            2.007902368169852,
            0.0036763097991141785,
        ],
        [
            ["I", "H", "Tx", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx", "Ty", "Tz", "Tx"],
            ["I", "H", "Ty", "Tx", "Ty", "Tz", "Tx", "Tz", "Tx", "Ty", "Tz", "Ty", "Tz"],
            5.179866016921518,
            0.004542822638048746,
        ],
    ],
)
def test_diamond_distance_for_mixed_diagonal(seq1, seq2, theta, distance):
    c = ch.ProbabilisticChannel.from_unitary_channels(
        ch.UnitaryChannel.from_sequence(seq1, True),
        ch.UnitaryChannel.from_sequence(seq2, True),
        theta,
        mc.NumpyConfig,
    )
    np.testing.assert_allclose(c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig), distance)


@pytest.mark.parametrize(
    ["proj_1", "seq_11", "seq_12", "proj_2", "seq_21", "seq_22", "theta", "distance"],
    [
        [
            ["I", "Z", "H", "S", "Tz", "Tx", "Ty", "Tx"],
            ["I", "Z", "I"],
            ["I", "S"],
            ["I", "S"],
            ["I", "Z", "I"],
            ["I", "S"],
            2.2336312299454693,
            0.033899366186582215,
        ],
        [
            ["I", "Z", "S"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            ["I", "Y", "H", "S", "H", "Tx", "Tz", "Ty", "Tx"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            3.978274738516034,
            0.021338488294120432,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.60506064948477,
            0.03282184847099254,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.552288085883322,
            0.022324504375577123,
        ],
        [
            ["I", "Y", "H", "S", "H", "Tx", "Tz", "Ty", "Tx", "Tz"],
            ["I", "I"],
            ["I", "Z", "S", "Tz"],
            ["I", "Z", "S"],
            ["I", "I"],
            ["I", "Z", "S"],
            0.7752283252787129,
            0.002603637233480823,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "S", "H", "Tz", "Tx", "Tz", "Tx", "Tz"],
            ["I", "Z", "I", "Tz"],
            ["I", "Z", "I"],
            2.369467427334453,
            0.0033169663527450343,
        ],
        [
            ["I", "Z", "S"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            ["I", "Y", "H", "S", "H", "Tx", "Ty", "Tz", "Tx"],
            ["I", "Z", "S"],
            ["I", "Z", "I"],
            0.8840386497791466,
            0.031826887148390826,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Ty", "Tx", "Ty"],
            ["I", "S"],
            ["I", "I"],
            2.4793681001683243,
            0.03392425218025496,
        ],
        [
            ["I", "S"],
            ["I", "S"],
            ["I", "I"],
            ["I", "X", "H", "S", "H", "Tz", "Tx", "Tz", "Ty"],
            ["I", "S"],
            ["I", "I"],
            5.594916529653181,
            0.03162395060972892,
        ],
    ],
)
def test_diamond_distance_for_mixed_fallback(
    proj_1, seq_11, seq_12, proj_2, seq_21, seq_22, theta, distance
):
    p1 = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj_1),
        correction=ch.ProbabilisticChannel.from_unitary_channels(
            ch.UnitaryChannel.from_sequence(seq_11, True),
            ch.UnitaryChannel.from_sequence(seq_12, True),
            theta,
            mc.NumpyConfig,
        ),
    )
    p2 = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(proj_2),
        correction=ch.ProbabilisticChannel.from_unitary_channels(
            ch.UnitaryChannel.from_sequence(seq_21, True),
            ch.UnitaryChannel.from_sequence(seq_22, True),
            theta,
            mc.NumpyConfig,
        ),
    )
    c = ch.ProbabilisticChannel.from_projective_channels(p1, p2, theta, mc.NumpyConfig)
    np.testing.assert_allclose(
        float(c.diamond_norm_distance_to_rz(theta, mc.NumpyConfig)), distance, atol=3e-5
    )


@pytest.mark.parametrize(
    "gates",
    [["I", "Z", "I", "Tz"], ["I", "S", "Tz"], ["I", "S", "Tz"], ["I", "Z", "S", "Tz"], ["I", "S"]],
)
@pytest.mark.parametrize("fmt", ["xz", "xyz"])
def test_unitary_to_cirq(gates, fmt):
    u = ch.UnitaryChannel.from_sequence(gates)
    assert u.to_cirq(fmt) == cirq.Circuit(ctr.to_cirq(u.to_matrix(), fmt))


@pytest.mark.parametrize(
    "gates1",
    [["I", "Z", "I", "Tz"], ["I", "S", "Tz"], ["I", "S", "Tz"], ["I", "Z", "S", "Tz"], ["I", "S"]],
)
@pytest.mark.parametrize(
    "gates2",
    [["I", "Z", "I", "Tz"], ["I", "S", "Tz"], ["I", "S", "Tz"], ["I", "Z", "S", "Tz"], ["I", "S"]],
)
@pytest.mark.parametrize("fmt", ["xz", "xyz"])
def test_fallback_to_cirq(gates1, gates2, fmt):
    c = ch.ProjectiveChannel(
        ch.UnitaryChannel.from_sequence(gates1), ch.UnitaryChannel.from_sequence(gates2)
    )
    q0, q1 = cirq.LineQubit.range(2)
    assert c.to_cirq(fmt) == cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.CircuitOperation(cirq.FrozenCircuit(ctr.to_cirq(c.rotation.to_matrix(), fmt, q0))),
        cirq.CNOT(q0, q1),
        cirq.measure(q1, key='m'),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(ctr.to_cirq(c.correction.to_matrix(), fmt, q0))  # type: ignore[attr-defined]
        ).with_classical_controls('m'),
    )
