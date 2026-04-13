# tests/span_counting_test.py
import pytest
import sympy
from qualtran import Signature

from qualtran._infra.composite_bloq import (
    BloqBuilder,
    Bloq,
    CompositeBloq,
)
from qualtran.resource_counting._costing import get_cost_value
from qualtran.bloqs.basic_gates import (
    CNOT,
    Hadamard,
    Toffoli,
    Swap,
)
from qualtran.bloqs.mcmt import And

from qualtran_flasq.span_counting import (
    BloqWithSpanInfo,
    GateSpan,
    TotalSpanCost,
    _calculate_spanning_distance,
    calculate_spans,
)


def test_bloq_with_span_basic():
    bloq = BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=3, compute_span=6)
    assert bloq.connect_span == 3
    assert bloq.compute_span == 6
    assert bloq.signature == CNOT().signature
    assert bloq.t_complexity() == CNOT().t_complexity()


def test_span_count_simple():
    bloq = BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=3, compute_span=3)
    assert get_cost_value(bloq, TotalSpanCost()) == GateSpan(
        connect_span=3, compute_span=3
    )

    # Test decomposition cost calculation
    bb = BloqBuilder()
    x = bb.add_register("x", 1)
    y = bb.add_register("y", 1)
    z = bb.add_register("z", 1)

    x = bb.add(Hadamard(), q=x)
    y = bb.add(Hadamard(), q=y)
    x, y = bb.add(CNOT(), ctrl=x, target=y)
    x, y = bb.add(CNOT(), ctrl=x, target=y)

    bloq = BloqWithSpanInfo(wrapped_bloq=CNOT(), connect_span=3, compute_span=3)
    x, y = bb.add(bloq, ctrl=x, target=y)
    x, y = bb.add(bloq, ctrl=x, target=y)

    ctrl_list, z = bb.add(Toffoli(), ctrl=[x, y], target=z)
    x = ctrl_list[0]
    y = ctrl_list[1]

    cbloq = bb.finalize(x=x, y=y, z=z)

    cost_val = get_cost_value(cbloq, TotalSpanCost())
    # Expect uncounted CNOT and Toffoli as they don't have inherent span info
    # when added directly to the builder without being wrapped.
    assert cost_val == GateSpan(
        connect_span=6, compute_span=6, uncounted_bloqs={CNOT(): 2, Toffoli(): 1}
    )


# --- New tests for new functions ---


@pytest.mark.parametrize(
    "coords, expected_distance",
    [
        ([(0, 0)], 0),
        ([(0, 0), (3, 4)], 7),  # 2-qubit
        ([(0, 0), (3, 4), (1, 5)], 8),  # 3-qubit
        ([(0, 0), (0, 0)], 0),  # 2-qubit, same location
    ],
)
def test_calculate_spanning_distance(coords, expected_distance):
    assert _calculate_spanning_distance(coords) == expected_distance


def test_calculate_spanning_distance_errors():
    with pytest.raises(NotImplementedError):
        _calculate_spanning_distance([(0, 0), (1, 1), (2, 2), (3, 3)])


def test_calculate_spans_validation():
    with pytest.raises(ValueError):
        calculate_spans(coords=[(0, 0)], bloq=CNOT())


def test_calculate_spans_single_qubit():
    connect_span, compute_span = calculate_spans(coords=[(0, 0)], bloq=Hadamard())
    assert connect_span == 0
    assert compute_span == 0


def test_calculate_spans_default_rule():
    # Toffoli uses the default rule
    coords = [(0, 0), (1, 1), (2, 0)]
    connect_span, compute_span = calculate_spans(coords=coords, bloq=Toffoli())
    spanning_distance = _calculate_spanning_distance(coords)  # 3
    # Default rule is now (D, D)
    assert connect_span == spanning_distance
    assert compute_span == spanning_distance


def test_calculate_spans_cnot_uses_default_rule():
    coords = [(0, 0), (3, 4)]
    connect_span, compute_span = calculate_spans(coords=coords, bloq=CNOT())
    spanning_distance = _calculate_spanning_distance(coords)  # 7
    # CNOT now uses the default (D, D) rule
    assert connect_span == spanning_distance
    assert compute_span == spanning_distance


def test_calculate_spans_and_uncompute_rule():
    coords = [(0, 0), (1, 1), (2, 2)]  # 3 qubits for And
    bloq = And(uncompute=True)
    connect_span, compute_span = calculate_spans(coords=coords, bloq=bloq)

    # For And(uncompute=True), distance is between the two control qubits. Cost is (D, D).
    control_coords = coords[:2]
    spanning_distance = _calculate_spanning_distance(control_coords)  # 2
    assert connect_span == spanning_distance
    assert compute_span == spanning_distance


def test_calculate_spans_swap_rule():
    coords = [(0, 0), (3, 4)]
    connect_span, compute_span = calculate_spans(coords=coords, bloq=Swap(1))
    spanning_distance = _calculate_spanning_distance(coords)  # 7
    # SWAP uses the default rule: connect=dist, compute=2*dist
    # This is now an explicit rule for Swap.
    assert connect_span == spanning_distance
    assert compute_span == 2 * spanning_distance


# =============================================================================
# Phase 1: Characterization tests for GateSpan arithmetic and TotalSpanCost
# =============================================================================


class TestGateSpanArithmetic:
    """Characterization tests for GateSpan __add__, __radd__, __mul__ edge cases."""

    def test_add_zero_returns_self(self):
        """GateSpan + 0 should return self (L83-84)."""
        gs = GateSpan(connect_span=3, compute_span=5)
        assert gs + 0 == gs

    def test_radd_zero_returns_self(self):
        """0 + GateSpan should return self (L100-101)."""
        gs = GateSpan(connect_span=3, compute_span=5)
        assert 0 + gs == gs

    def test_add_wrong_type_raises(self):
        """Adding a non-GateSpan, non-zero value should raise TypeError (L85-86)."""
        gs = GateSpan(connect_span=1, compute_span=1)
        with pytest.raises(TypeError, match="Can only add"):
            gs + "not_a_span"

    def test_mul_by_int(self):
        """GateSpan * int should scale spans and uncounted bloqs (L104-120)."""
        gs = GateSpan(connect_span=3, compute_span=5, uncounted_bloqs={CNOT(): 2})
        result = gs * 4
        assert result.connect_span == 12
        assert result.compute_span == 20
        assert result.uncounted_bloqs == {CNOT(): 8}

    def test_rmul_by_int(self):
        """int * GateSpan should also work (L122-123)."""
        gs = GateSpan(connect_span=3, compute_span=5)
        assert 4 * gs == gs * 4

    def test_mul_wrong_type_raises(self):
        """Multiplying by a non-numeric type should raise TypeError (L105-107)."""
        gs = GateSpan(connect_span=1, compute_span=1)
        with pytest.raises(TypeError, match="Can only multiply"):
            gs * "bad"

    def test_mul_by_sympy_expr(self):
        """Multiplying by a sympy expression should work."""
        gs = GateSpan(connect_span=2, compute_span=3)
        x = sympy.Symbol("x")
        result = gs * x
        assert result.connect_span == 2 * x
        assert result.compute_span == 3 * x


class TestGateSpanStringAndDict:
    """Characterization tests for GateSpan __str__ and asdict."""

    def test_str_empty_is_dash(self):
        """Empty GateSpan str should return '-' (L143-144)."""
        gs = GateSpan()
        assert str(gs) == "-"

    def test_str_with_spans(self):
        """Non-empty GateSpan should display spans."""
        gs = GateSpan(connect_span=3, compute_span=5)
        s = str(gs)
        assert "connect_span: 3" in s
        assert "compute_span: 5" in s

    def test_asdict_filters_zeros(self):
        """asdict should filter out zero spans and empty uncounted_bloqs (L148-156)."""
        gs = GateSpan()
        assert gs.asdict() == {}

    def test_asdict_with_values(self):
        gs = GateSpan(connect_span=3, compute_span=5)
        d = gs.asdict()
        assert d == {"connect_span": 3, "compute_span": 5}

    def test_asdict_with_uncounted_bloqs(self):
        gs = GateSpan(connect_span=0, compute_span=0, uncounted_bloqs={CNOT(): 1})
        d = gs.asdict()
        assert "uncounted_bloqs" in d
        assert d["uncounted_bloqs"] == {CNOT(): 1}


class TestTotalSpanCostValidation:
    """Characterization tests for TotalSpanCost edge cases."""

    def test_validate_val_wrong_type_raises(self):
        """validate_val should raise TypeError for non-GateSpan values (L286-288)."""
        cost_key = TotalSpanCost()
        with pytest.raises(TypeError, match="GateSpan"):
            cost_key.validate_val("not_a_gate_span")

    def test_str(self):
        """TotalSpanCost string representation (L291-292)."""
        assert str(TotalSpanCost()) == "total span cost"
