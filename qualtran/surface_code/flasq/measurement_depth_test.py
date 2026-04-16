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

import attrs

# test_measurement_depth.py
import pytest
import sympy
from frozendict import frozendict

# Imports from Qualtran for test setup
from qualtran import (
    Bloq,
    BloqBuilder,
    CompositeBloq,
    DecomposeNotImplementedError,
    QUInt,
    Signature,
)
from qualtran.bloqs.arithmetic import Add, HammingWeightCompute
from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate
from qualtran.bloqs.mcmt import And
from qualtran.resource_counting import CostKey, get_cost_value

# Imports from the module being tested
from qualtran.surface_code.flasq.measurement_depth import (
    MeasurementDepth,
    TotalMeasurementDepth,
)

# --- Helper Bloqs for Testing ---


@attrs.frozen(kw_only=True)
class BloqWithStaticMeasurementDepth(Bloq):
    """Wraps a bloq to add a static MeasurementDepth cost for testing."""

    wrapped_bloq: Bloq
    measurement_depth_cost: MeasurementDepth = attrs.field(factory=MeasurementDepth)

    @property
    def signature(self) -> Signature:
        return self.wrapped_bloq.signature

    def build_composite_bloq(self, bb: BloqBuilder, **soqs):
        return bb.add_d(self.wrapped_bloq, **soqs)

    def my_static_costs(self, cost_key: CostKey):
        if isinstance(cost_key, TotalMeasurementDepth):
            return self.measurement_depth_cost
        return NotImplemented

    def __str__(self):
        return f"StaticDepthWrapper({self.wrapped_bloq}, depth_cost={self.measurement_depth_cost})"


@attrs.frozen(kw_only=True)
class UnknownBloq(Bloq):
    """Dummy Bloq that cannot be decomposed for cost analysis."""

    @property
    def signature(self) -> Signature:
        return Signature.build(q=1)

    def decompose_bloq(self) -> "CompositeBloq":
        raise DecomposeNotImplementedError(f"{self} is atomic.")


# --- Tests for MeasurementDepth Data Class ---


def test_measurement_depth_init():
    md_default = MeasurementDepth()
    assert md_default.depth == 0
    assert md_default.bloqs_with_unknown_depth == frozendict()
    assert isinstance(md_default.bloqs_with_unknown_depth, frozendict)

    md_val = MeasurementDepth(depth=5, bloqs_with_unknown_depth={TGate(): 2})
    assert md_val.depth == 5
    assert md_val.bloqs_with_unknown_depth == {TGate(): 2}
    assert isinstance(md_val.bloqs_with_unknown_depth, frozendict)


def test_measurement_depth_add():
    md1 = MeasurementDepth(depth=3, bloqs_with_unknown_depth={TGate(): 1, CNOT(): 1})
    md2 = MeasurementDepth(
        depth=5, bloqs_with_unknown_depth={TGate(): 2, Hadamard(): 3}
    )
    md_sum = md1 + md2
    assert md_sum.depth == 8
    assert md_sum.bloqs_with_unknown_depth == {TGate(): 3, CNOT(): 1, Hadamard(): 3}
    assert isinstance(md_sum.bloqs_with_unknown_depth, frozendict)

    # Test adding zero identity
    assert md1 + 0 == md1
    assert 0 + md1 == md1

    # Test type error on invalid addition
    with pytest.raises(TypeError):
        _ = md1 + "string"


def test_measurement_depth_str():
    md0 = MeasurementDepth()
    assert str(md0) == "MeasurementDepth(depth: 0)"

    md1 = MeasurementDepth(depth=5)
    assert str(md1) == "MeasurementDepth(depth: 5)"

    md2 = MeasurementDepth(bloqs_with_unknown_depth={TGate(): 3})
    assert str(md2) == "MeasurementDepth(bloqs_with_unknown_depth: {T: 3})"

    md3 = MeasurementDepth(
        depth=sympy.Symbol("d"), bloqs_with_unknown_depth={CNOT(): 1, TGate(): 2}
    )
    # Keys should be sorted alphabetically by string representation in the output
    assert (
        str(md3)
        == "MeasurementDepth(bloqs_with_unknown_depth: {CNOT: 1, T: 2}, depth: d)"
    )


def test_measurement_depth_asdict():
    md0 = MeasurementDepth()
    assert md0.asdict() == {}  # Zero depth and empty dict are filtered

    md1 = MeasurementDepth(depth=5)
    assert md1.asdict() == {"depth": 5}

    md2 = MeasurementDepth(bloqs_with_unknown_depth={TGate(): 3})
    assert md2.asdict() == {"bloqs_with_unknown_depth": {TGate(): 3}}
    assert isinstance(md2.asdict()["bloqs_with_unknown_depth"], frozendict)

    md3 = MeasurementDepth(
        depth=sympy.Symbol("d"), bloqs_with_unknown_depth={CNOT(): 1, TGate(): 2}
    )
    expected_dict3 = {
        "depth": sympy.Symbol("d"),
        "bloqs_with_unknown_depth": frozendict({CNOT(): 1, TGate(): 2}),
    }
    assert md3.asdict() == expected_dict3
    assert isinstance(md3.asdict()["bloqs_with_unknown_depth"], frozendict)

    md4 = MeasurementDepth(depth=0, bloqs_with_unknown_depth={TGate(): 1})
    expected_dict4 = {"bloqs_with_unknown_depth": frozendict({TGate(): 1})}
    assert md4.asdict() == expected_dict4  # Zero depth is filtered
    assert isinstance(md4.asdict()["bloqs_with_unknown_depth"], frozendict)


# --- Tests for TotalMeasurementDepth Cost Key ---


def test_total_measurement_depth_zero():
    cost_key = TotalMeasurementDepth()
    assert cost_key.zero() == MeasurementDepth(depth=0)


def test_total_measurement_depth_compute_static():
    """Test retrieving cost directly via my_static_costs."""
    cost_key = TotalMeasurementDepth()
    static_cost = MeasurementDepth(depth=5)
    bloq = BloqWithStaticMeasurementDepth(
        wrapped_bloq=TGate(), measurement_depth_cost=static_cost
    )
    assert get_cost_value(bloq, cost_key) == static_cost


def test_total_measurement_depth_compute_unknown():
    """Test the fallback case for an unknown/atomic bloq."""
    cost_key = TotalMeasurementDepth()
    bloq = UnknownBloq()
    expected = MeasurementDepth(depth=0, bloqs_with_unknown_depth={bloq: 1})
    result = get_cost_value(bloq, cost_key)
    assert result == expected
    assert isinstance(result.bloqs_with_unknown_depth, frozendict)


def test_total_measurement_depth_compute_composite_simple_serial():
    """Test depth calculation for a simple sequential composite bloq."""
    cost_key = TotalMeasurementDepth()
    bb = BloqBuilder()
    q1 = bb.add_register("q1", 1)
    q2 = bb.add_register("q2", 1)

    q1 = bb.add(Hadamard(), q=q1)  # Depth 0
    (q1, q2), q3 = bb.add(And(), ctrl=[q1, q2])  # Depth 1
    q2 = bb.add(Hadamard(), q=q2)  # Depth 0

    cbloq = bb.finalize(q1=q1, q2=q2, q3=q3)
    # Longest path: H -> And -> H = 0 + 1 + 0 = 1
    expected_depth = MeasurementDepth(depth=1)
    assert get_cost_value(cbloq, cost_key) == expected_depth


def test_total_measurement_depth_compute_composite_parallel():
    """Test depth calculation involving parallel and serial paths."""
    cost_key = TotalMeasurementDepth()
    bb = BloqBuilder()
    q1 = bb.add_register("q1", 1)
    q2 = bb.add_register("q2", 1)
    q4 = bb.add_register("q4", 1)
    q5 = bb.add_register("q5", 1)

    q1 = bb.add(Hadamard(), q=q1)  # Depth 0
    (q1, q2), q3 = bb.add(And(), ctrl=[q1, q2])  # Depth 1 (Path 1)
    (q4, q5), q6 = bb.add(And(), ctrl=[q4, q5])  # Depth 1 (Path 2, parallel to Path 1)
    (q2, q4), q7 = bb.add(
        And(), ctrl=[q2, q4]
    )  # Depth 1 (Depends on outputs of both previous Ands)
    q2 = bb.add(Hadamard(), q=q2)  # Depth 0

    cbloq = bb.finalize(q1=q1, q2=q2, q3=q3, q4=q4, q5=q5, q6=q6, q7=q7)

    # Longest path: H -> And(q1,q2) -> And(q2,q4) -> H = 0 + 1 + 1 + 0 = 2
    # Other path: And(q4,q5) -> And(q2,q4) -> H = 1 + 1 + 0 = 2
    expected_depth = MeasurementDepth(depth=2)
    assert get_cost_value(cbloq, cost_key) == expected_depth


def test_total_measurement_depth_compute_adder():
    """Test depth calculation for the Add bloq."""
    bitsize = 5
    adder_bloq = Add(QUInt(bitsize))

    calculated_depth = get_cost_value(adder_bloq, TotalMeasurementDepth())

    # Check structure of the result
    assert isinstance(calculated_depth, MeasurementDepth)
    assert isinstance(calculated_depth.bloqs_with_unknown_depth, frozendict)
    assert (
        not calculated_depth.bloqs_with_unknown_depth
    )  # Expect Add to decompose fully

    assert calculated_depth.depth == 8  # Checked by inspection of the reference paper.


def test_total_measurement_depth_with_rotation_depth():
    """Test TotalMeasurementDepth when rotation_depth is specified."""
    custom_rotation_depth = 4.5
    cost_key = TotalMeasurementDepth(rotation_depth=custom_rotation_depth)

    # Test individual rotation gates
    from qualtran.bloqs.basic_gates import Rx, Rz, XPowGate, ZPowGate  # type: ignore[attr-defined]

    rx_bloq = Rx(angle=sympy.Symbol("theta_rx"))
    rz_bloq = Rz(angle=sympy.Symbol("theta_rz"))
    xpow_bloq = XPowGate(exponent=sympy.Symbol("exp_x"))
    zpow_bloq = ZPowGate(exponent=sympy.Symbol("exp_z"))

    expected_rotation_cost = MeasurementDepth(depth=custom_rotation_depth)

    assert get_cost_value(rx_bloq, cost_key) == expected_rotation_cost
    assert get_cost_value(rz_bloq, cost_key) == expected_rotation_cost
    assert get_cost_value(xpow_bloq, cost_key) == expected_rotation_cost
    assert get_cost_value(zpow_bloq, cost_key) == expected_rotation_cost

    # Test that non-rotation gates are unaffected
    assert get_cost_value(TGate(), cost_key) == MeasurementDepth(depth=1)
    assert get_cost_value(Hadamard(), cost_key) == MeasurementDepth(depth=0)
    assert get_cost_value(CNOT(), cost_key) == MeasurementDepth(depth=0)

    # Test a composite bloq with a rotation
    bb = BloqBuilder()
    q1 = bb.add_register("q1", 1)
    q1 = bb.add(Hadamard(), q=q1)  # Depth 0
    q1 = bb.add(rx_bloq, q=q1)  # Depth custom_rotation_depth (0.5)
    q1 = bb.add(TGate(), q=q1)  # Depth 1
    cbloq_with_rotation = bb.finalize(q1=q1)

    # Longest path: H (0) -> Rx (4.5) -> T (1) = 0 + 0.5 + 1 = 5.5
    expected_composite_depth = MeasurementDepth(depth=5.5)
    assert get_cost_value(cbloq_with_rotation, cost_key) == expected_composite_depth

    # Test that if rotation_depth is None (default), rotations are unknown
    default_cost_key = TotalMeasurementDepth()  # rotation_depth is None
    assert get_cost_value(rx_bloq, default_cost_key) == MeasurementDepth(
        bloqs_with_unknown_depth={rx_bloq: 1}
    )
    # And T gate is still 1
    assert get_cost_value(TGate(), default_cost_key) == MeasurementDepth(depth=1)

    # Test that if rotation_depth is 0, rotations are free
    zero_rotation_depth_key = TotalMeasurementDepth(rotation_depth=0)
    assert get_cost_value(rx_bloq, zero_rotation_depth_key) == MeasurementDepth(depth=0)


# =============================================================================
# Phase 1: Characterization tests for untested measurement_depth branches
# =============================================================================

from qualtran.bloqs.basic_gates import OneState, ZeroEffect, ZeroState


class TotalMeasurementDepthBaseCasesTestSuite:
    """Characterization tests for base cases in TotalMeasurementDepth.compute."""

    def test_state_or_effect_depth_zero(self):
        """ZeroState, OneState, ZeroEffect should have depth 0 (L259-260)."""
        cost_key = TotalMeasurementDepth()
        assert get_cost_value(ZeroState(), cost_key) == MeasurementDepth(depth=0)
        assert get_cost_value(OneState(), cost_key) == MeasurementDepth(depth=0)
        assert get_cost_value(ZeroEffect(), cost_key) == MeasurementDepth(depth=0)


class TotalMeasurementDepthEdgeCasesTestSuite:
    """Characterization tests for TotalMeasurementDepth edge cases."""

    def test_validate_val_wrong_type_raises(self):
        """validate_val should raise TypeError for non-MeasurementDepth values (L310-312)."""
        cost_key = TotalMeasurementDepth()
        with pytest.raises(TypeError, match="MeasurementDepth"):
            cost_key.validate_val("not_a_measurement_depth")

    def test_str(self):
        """TotalMeasurementDepth string representation (L316-317)."""
        assert str(TotalMeasurementDepth()) == "total measurement depth"

    def test_radd_nonzero_returns_not_implemented(self):
        """MeasurementDepth.__radd__ with non-zero non-MeasurementDepth delegates to __add__ (L95-96)."""
        md = MeasurementDepth(depth=5)
        # __add__ returns NotImplemented for non-MeasurementDepth, non-zero
        result = md.__add__("bad")
        assert result is NotImplemented


# NOTE: The HammingWeightCompute test was previously commented out with placeholder
# expected values (222). The correct expected depths depend on the specific
# decomposition of HammingWeightCompute, which may vary across Qualtran versions.
# Rather than guessing values, this is deferred to Phase 3 where we can verify
# against the paper's expectations. The commented-out code has been removed as
# part of Phase 1 cleanup.
