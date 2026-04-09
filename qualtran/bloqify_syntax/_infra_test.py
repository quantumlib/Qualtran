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
import inspect
from typing import Dict

import pytest

import qualtran as qlt
from qualtran import BloqBuilder, QVar


def func1(x, k, n=1):
    pass


def call_func1_with_kwargs(*args, **kwargs):
    stuff = inspect.signature(func1).bind(*args, **kwargs).arguments
    return dict(stuff)


def test_inspect():
    a = call_func1_with_kwargs('xv', 'kv', n=2)
    assert a == {'x': 'xv', 'k': 'kv', 'n': 2}

    # Note: we could use `.apply_defaults` if we wanted the default values
    a = call_func1_with_kwargs('xv', 'kv')
    assert a == {'x': 'xv', 'k': 'kv'}

    a = call_func1_with_kwargs('xv', n=2, k=1)
    assert a == {'x': 'xv', 'n': 2, 'k': 1}


@qlt.bloqify
def minimal_bloq(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
    return {'x': x}


def test_bloqify_decorator():
    assert minimal_bloq.name == 'minimal_bloq'
    assert minimal_bloq.pkg == 'qualtran.bloqify_syntax._infra_test'


def test_bloqify_make():
    sig = qlt.Signature.build(x=1)
    cbloq = minimal_bloq.make(sig)
    assert isinstance(cbloq, qlt.CompositeBloq)
    assert cbloq.signature == sig


def test_bloqify_call():
    @qlt.bloqify
    def outer_program(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
        x = minimal_bloq(bb, x=x)
        return {'x': x}

    sig = qlt.Signature.build(x=1)
    cbloq = outer_program.make(sig)
    assert isinstance(cbloq, qlt.CompositeBloq)


def test_bloqify_inline():
    @qlt.bloqify
    def outer_program(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
        x_out = minimal_bloq.inline(bb, x=x)
        return {'x': x_out[0]}

    sig = qlt.Signature.build(x=1)
    cbloq = outer_program.make(sig)
    assert isinstance(cbloq, qlt.CompositeBloq)


def test_bloqify_l1():
    sig = qlt.Signature.build(x=1)
    l1_str = minimal_bloq.dump_l1(sig)
    assert isinstance(l1_str, str)


def test_bloqify_non_dict_return():
    @qlt.bloqify
    def bad_bloq(bb: 'BloqBuilder', x: 'QVar'):
        return x  # Not a dict

    @qlt.bloqify
    def outer_program(bb: 'BloqBuilder', x: 'QVar') -> Dict[str, 'QVar']:
        x = bad_bloq(bb, x=x)
        return {'x': x}

    sig = qlt.Signature.build(x=1)
    with pytest.raises(ValueError, match="bad_bloq is expected to return a dictionary"):
        outer_program.make(sig)


def test_bloqify_make_shadowing():
    sig = qlt.Signature.build(x=1)
    with pytest.raises(ValueError, match="shadow quantum register names"):
        minimal_bloq.make(sig, x=1)


def test_bloqify_missing_bb():
    with pytest.raises(ValueError, match="must take 'bb' as its first argument"):

        @qlt.bloqify  # type: ignore[arg-type]
        def no_bb_func(x: 'QVar'):
            return {'x': x}
