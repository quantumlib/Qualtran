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
from typing import Callable, Optional, Type

from attrs import field, frozen

from .bloq import Bloq


@frozen
class BloqExample:
    _func: Callable[[], Bloq] = field(repr=False, hash=False)
    name: str
    bloq_cls: Type[Bloq]
    generalizer: Callable[[Bloq], Optional[Bloq]] = lambda x: x

    def make(self) -> Bloq:
        """Make the bloq."""
        return self._func()

    def __call__(self) -> Bloq:
        """This class is callable: it will make the bloq.

        This makes the `bloq_example` decorator make sense: we wrap a function, so this
        callable should do the same thing when called.
        """
        return self.make()


def _name_from_func_name(func: Callable[[], Bloq]):
    return func.__name__.lstrip('_')


def _bloq_cls_from_func_annotation(func: Callable[[], Bloq]):
    anno = func.__annotations__
    if 'return' not in anno:
        raise ValueError(f'{func} must have a return type annotation.')

    cls = anno['return']
    assert issubclass(cls, Bloq), cls
    return cls


def bloq_example(
    _func: Callable[[], Bloq] = None, *, generalizer: Callable[[Bloq], Optional[Bloq]] = lambda x: x
):
    """Decorator to turn a function into a `BloqExample`.

    This will set `name` to the name of the function and `bloq_cls` according to the return-type
    annotation. You can also call the decorator with keyword arguments, which will be passed
    through to the `BloqExample` constructor.
    """

    def _inner(func: Callable[[], Bloq]) -> BloqExample:
        return BloqExample(
            func=func,
            name=_name_from_func_name(func),
            bloq_cls=_bloq_cls_from_func_annotation(func),
            generalizer=generalizer,
        )

    if _func is None:
        return _inner
    return _inner(_func)
