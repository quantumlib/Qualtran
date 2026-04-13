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

# test_combined_counting.py
import numpy as np
from qualtran import Signature

from qualtran._infra.composite_bloq import (
    BloqBuilder,
    Bloq,
)
from qualtran.resource_counting._costing import get_cost_value
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    Toffoli,
)
from qualtran.cirq_interop import CirqGateAsBloq


from qualtran.surface_code.flasq.span_counting import BloqWithSpanInfo
from qualtran.surface_code.flasq.volume_counting import (
    FLASQGateCounts,
    FLASQGateTotals,
)
from attrs import frozen
import cirq


def test_flasq_count_basic():
    bloq = CNOT()
    cost_val = get_cost_value(bloq, FLASQGateTotals())
    assert cost_val == FLASQGateCounts(cnot=1)

    bloq_h = Hadamard()
    cost_val_h = get_cost_value(bloq_h, FLASQGateTotals())
    assert cost_val_h == FLASQGateCounts(hadamard=1)

    bloq_t = Toffoli()
    cost_val_t = get_cost_value(bloq_t, FLASQGateTotals())
    assert cost_val_t == FLASQGateCounts(toffoli=1)


def test_flasq_count_wrapped_bloq():
    # FLASQ counts should ignore the span wrapper and count the inner bloq
    bloq = BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=3, compute_span=3)
    cost_val = get_cost_value(bloq, FLASQGateTotals())
    assert cost_val == FLASQGateCounts(cnot=1)
    bloq_h_wrapped = BloqWithSpanInfo(
        wrapped_bloq=Hadamard(), connect_span=0, compute_span=0
    )
    cost_val_h = get_cost_value(bloq_h_wrapped, FLASQGateTotals())
    assert cost_val_h == FLASQGateCounts(hadamard=1)


def test_flasq_count_composite():
    bb = BloqBuilder()
    q0 = bb.add_register("q0", 1)
    q1 = bb.add_register("q1", 1)

    q0 = bb.add(Hadamard(), q=q0)
    q0, q1 = bb.add(CNOT(), ctrl=q0, target=q1)
    q1 = bb.add(Hadamard(), q=q1)
    # Finalize using the *latest* soquets
    cbloq = bb.finalize(q0=q0, q1=q1)

    cost_val = get_cost_value(cbloq, FLASQGateTotals())
    assert cost_val == FLASQGateCounts(hadamard=2, cnot=1)


def test_flasq_count_unknown():
    # Create a dummy bloq with no decomposition or base case for FLASQ counting
    @frozen(kw_only=True)
    class UnknownBloq(Bloq):
        @property
        def signature(self) -> Signature:
            return Signature.build(q=1)

    bloq = UnknownBloq()
    cost_val = get_cost_value(bloq, FLASQGateTotals())
    # Expect the coster to record the bloq as unknown
    assert cost_val == FLASQGateCounts(bloqs_with_unknown_cost={bloq: 1})


def test_flasq_counts_str_with_pow_gates():
    """Tests the __str__ and asdict methods with CirqGateAsBloq(*PowGate) keys.

    This simulates the scenario where XXZ model gates are marked as unknown
    and checks if printing or converting to dict causes the unhashable TypeError.
    """
    # Simulate exponents similar to those from the XXZ model
    j, dt, delta = 1.0, 0.1, 0.5
    exponent_xy = (j * dt) / np.pi
    exponent_zz = (j * delta * dt) / np.pi
    xx_gate = cirq.XXPowGate(exponent=exponent_xy, global_shift=-0.5)
    yy_gate = cirq.YYPowGate(exponent=exponent_xy, global_shift=-0.5)
    zz_gate = cirq.ZZPowGate(exponent=exponent_zz, global_shift=-0.5)
    xx_bloq = CirqGateAsBloq(xx_gate)
    yy_bloq = CirqGateAsBloq(yy_gate)
    zz_bloq = CirqGateAsBloq(zz_gate)
    counts_with_unknown = FLASQGateCounts(
        t=10,
        cnot=5,
        bloqs_with_unknown_cost={
            xx_bloq: 8,
            yy_bloq: 8,
            zz_bloq: 8,
        },
    )
    # 1. Test calling str() - this triggers the error reported in the traceback
    counts_str = str(counts_with_unknown)
    # If successful, perform basic checks on the output string

    assert "t: 10" in counts_str
    assert "cnot: 5" in counts_str
    assert "bloqs_with_unknown_cost" in counts_str
    assert "cirq.XX**" in counts_str
    assert "cirq.YY**" in counts_str
    assert "cirq.ZZ**" in counts_str
    assert ": 8" in counts_str
    # 2. Test calling asdict() directly - this is called by __str__
    counts_dict = counts_with_unknown.asdict()
    # If successful, perform basic checks on the output dict

    assert "t" in counts_dict and counts_dict["t"] == 10
    assert "cnot" in counts_dict and counts_dict["cnot"] == 5
    assert "bloqs_with_unknown_cost" in counts_dict
    unknown_dict = counts_dict["bloqs_with_unknown_cost"]
    assert isinstance(unknown_dict, dict)  # Should be dict, not frozendict after asdict
    assert xx_bloq in unknown_dict and unknown_dict[xx_bloq] == 8
    assert yy_bloq in unknown_dict and unknown_dict[yy_bloq] == 8
    assert zz_bloq in unknown_dict and unknown_dict[zz_bloq] == 8


# =============================================================================
# Phase 1: Characterization tests for untested branches in FLASQGateTotals.compute
# =============================================================================

import pytest
from qualtran.bloqs.basic_gates import (
    SGate,
    CZ,
    XGate,
    YGate,
    ZGate,
    ZPowGate,
    XPowGate,
    YPowGate,
    Rz,
    Rx,
)
from qualtran.bloqs.basic_gates.z_basis import MeasureZ
from qualtran.bloqs.basic_gates.x_basis import MeasureX
from qualtran.bloqs.basic_gates.global_phase import GlobalPhase
from qualtran.bloqs.basic_gates.identity import Identity
from qualtran.bloqs.basic_gates import ZeroState, OneState, ZeroEffect
from qualtran.bloqs.mcmt import And


class FLASQGateTotalsBaseCasesTestSuite:
    """Characterization tests for each branch in FLASQGateTotals.compute."""

    # --- Pauli gates are free ---

    def test_x_gate_free(self):
        assert get_cost_value(XGate(), FLASQGateTotals()) == FLASQGateCounts()

    def test_y_gate_free(self):
        assert get_cost_value(YGate(), FLASQGateTotals()) == FLASQGateCounts()

    def test_z_gate_free(self):
        assert get_cost_value(ZGate(), FLASQGateTotals()) == FLASQGateCounts()

    # --- PowGates at identity (exponent=0 or 1) are free ---

    def test_zpow_identity_free(self):
        assert get_cost_value(ZPowGate(exponent=0.0), FLASQGateTotals()) == FLASQGateCounts()
        assert get_cost_value(ZPowGate(exponent=1.0), FLASQGateTotals()) == FLASQGateCounts()

    def test_xpow_identity_free(self):
        assert get_cost_value(XPowGate(exponent=0.0), FLASQGateTotals()) == FLASQGateCounts()
        assert get_cost_value(XPowGate(exponent=1.0), FLASQGateTotals()) == FLASQGateCounts()

    def test_ypow_identity_free(self):
        assert get_cost_value(YPowGate(exponent=0.0), FLASQGateTotals()) == FLASQGateCounts()
        assert get_cost_value(YPowGate(exponent=1.0), FLASQGateTotals()) == FLASQGateCounts()

    # --- SGate, CZ ---

    def test_s_gate(self):
        assert get_cost_value(SGate(), FLASQGateTotals()) == FLASQGateCounts(s_gate=1)

    def test_cz_gate(self):
        assert get_cost_value(CZ(), FLASQGateTotals()) == FLASQGateCounts(cz=1)

    # --- And gates ---

    def test_and_gate_compute(self):
        assert get_cost_value(And(), FLASQGateTotals()) == FLASQGateCounts(and_gate=1)

    def test_and_gate_uncompute(self):
        assert get_cost_value(And(uncompute=True), FLASQGateTotals()) == FLASQGateCounts(and_dagger_gate=1)

    # --- Measurements are free ---

    def test_measure_z_free(self):
        assert get_cost_value(MeasureZ(), FLASQGateTotals()) == FLASQGateCounts()

    def test_measure_x_free(self):
        assert get_cost_value(MeasureX(), FLASQGateTotals()) == FLASQGateCounts()

    # --- Bookkeeping and special ---

    def test_global_phase_free(self):
        assert get_cost_value(GlobalPhase(exponent=0.5), FLASQGateTotals()) == FLASQGateCounts()

    def test_identity_free(self):
        assert get_cost_value(Identity(), FLASQGateTotals()) == FLASQGateCounts()

    # --- States and effects are free ---

    def test_zero_state_free(self):
        assert get_cost_value(ZeroState(), FLASQGateTotals()) == FLASQGateCounts()

    def test_one_state_free(self):
        assert get_cost_value(OneState(), FLASQGateTotals()) == FLASQGateCounts()

    def test_zero_effect_free(self):
        assert get_cost_value(ZeroEffect(), FLASQGateTotals()) == FLASQGateCounts()

    # --- PowGate at sqrt (±0.5) -> decomposed as Clifford ---

    def test_zpow_half_is_s(self):
        """ZPowGate(0.5) = sqrt(Z) = S gate."""
        assert get_cost_value(ZPowGate(exponent=0.5), FLASQGateTotals()) == FLASQGateCounts(s_gate=1)
        assert get_cost_value(ZPowGate(exponent=-0.5), FLASQGateTotals()) == FLASQGateCounts(s_gate=1)

    def test_xpow_half_is_hsh(self):
        """XPowGate(0.5) = sqrt(X) = H S H."""
        assert get_cost_value(XPowGate(exponent=0.5), FLASQGateTotals()) == FLASQGateCounts(hadamard=2, s_gate=1)

    def test_ypow_half_is_h(self):
        """YPowGate(0.5) = sqrt(Y), proportional to XH."""
        assert get_cost_value(YPowGate(exponent=0.5), FLASQGateTotals()) == FLASQGateCounts(hadamard=1)

    # --- Arbitrary rotations ---

    def test_rz_rotation(self):
        assert get_cost_value(Rz(angle=0.123), FLASQGateTotals()) == FLASQGateCounts(z_rotation=1)

    def test_rx_rotation(self):
        assert get_cost_value(Rx(angle=0.123), FLASQGateTotals()) == FLASQGateCounts(x_rotation=1)

    def test_zpow_arbitrary_rotation(self):
        """ZPowGate at a non-Clifford angle should count as z_rotation."""
        result = get_cost_value(ZPowGate(exponent=0.123), FLASQGateTotals())
        assert result == FLASQGateCounts(z_rotation=1)

    def test_xpow_arbitrary_rotation(self):
        """XPowGate at a non-Clifford angle should count as x_rotation."""
        result = get_cost_value(XPowGate(exponent=0.123), FLASQGateTotals())
        assert result == FLASQGateCounts(x_rotation=1)

    # --- ZZPowGate decomposition ---

    def test_zzpow_gate_counts(self):
        """CirqGateAsBloq(ZZPowGate) should decompose to 2 CNOTs + 1 z_rotation."""
        bloq = CirqGateAsBloq(cirq.ZZPowGate(exponent=0.1))
        assert get_cost_value(bloq, FLASQGateTotals()) == FLASQGateCounts(cnot=2, z_rotation=1)

    # --- Cirq measurement/reset operations are free ---

    def test_cirq_measurement_free(self):
        bloq = CirqGateAsBloq(cirq.MeasurementGate(num_qubits=1, key="m"))
        assert get_cost_value(bloq, FLASQGateTotals()) == FLASQGateCounts()

    def test_cirq_reset_free(self):
        bloq = CirqGateAsBloq(cirq.ResetChannel())
        assert get_cost_value(bloq, FLASQGateTotals()) == FLASQGateCounts()


class FLASQGateCountsArithmeticTestSuite:
    """Characterization tests for FLASQGateCounts arithmetic error handling."""

    def test_add_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Can only add"):
            FLASQGateCounts(t=1) + "not_a_count"

    def test_mul_wrong_type_raises(self):
        with pytest.raises(TypeError, match="Can only multiply"):
            FLASQGateCounts(t=1) * "not_a_number"

    def test_add_zero_returns_self(self):
        counts = FLASQGateCounts(t=5, cnot=3)
        assert counts + 0 == counts
        assert 0 + counts == counts

    def test_mul_by_int(self):
        counts = FLASQGateCounts(t=2, cnot=3)
        result = counts * 5
        assert result == FLASQGateCounts(t=10, cnot=15)

    def test_total_rotations_property(self):
        counts = FLASQGateCounts(x_rotation=3, z_rotation=7)
        assert counts.total_rotations == 10

