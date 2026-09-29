from __future__ import annotations

import pytest

import kickmix as km


def test_circuit():
    with pytest.raises(ValueError, match="unknown operation name"):
        km.Circuit("test")
    c = km.Circuit("X q0")
    assert len(c) == 1
