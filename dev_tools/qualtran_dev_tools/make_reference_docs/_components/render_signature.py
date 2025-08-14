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
import re
from typing import Optional

from griffe import Function, Object, Parameter, ParameterKind

BESPOKE_OBJECT_NAMES_FOR_CLASSES = {'BloqBuilder': 'bb', 'CompositeBloq': 'cbloq'}


def format_parameter(p: Parameter) -> str:
    if p.annotation is not None and p.default is not None:
        s = f"{p.name}: {p.annotation} = {p.default}"
    elif p.annotation is not None and p.default is None:
        s = f"{p.name}: {p.annotation}"
    elif p.annotation is None and p.default is not None:
        s = f"{p.name} = {p.default}"
    else:
        s = f"{p.name}"

    if p.kind and p.kind is ParameterKind.positional_or_keyword:
        return s
    elif p.kind and p.kind is ParameterKind.positional_only:
        return s  # TODO?
    elif p.kind and p.kind is ParameterKind.keyword_only:
        return s  # TODO?
    elif p.kind and p.kind is ParameterKind.var_positional:
        return f'*{s}'
    elif p.kind and p.kind is ParameterKind.var_keyword:
        return f'**{s}'
    else:
        raise ValueError(p.kind)


def camel_to_snake(name):
    # Turn an UpperCamelCaseName into lower_snake_case
    # .. we want to make it look like we're calling methods on instances of the class.
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return s1.lower()


def get_obj_instance_name(obj):
    # Get prototypical object instance names for a given class.

    # First, check if we have a bespoke name
    n = BESPOKE_OBJECT_NAMES_FOR_CLASSES.get(obj.name, None)
    if n is not None:
        return n

    # Otherwise, use the class name as lower_snake_case.
    return camel_to_snake(obj.name)


def write_property_method_signature(f, obj, obj2: Function):
    assert obj2.returns, obj2
    obj_instance_name = get_obj_instance_name(obj)
    method_signature = f'{obj_instance_name}.{obj2.name} -> {obj2.returns}'
    f.write(f'```python\n{method_signature}\n```\n\n')


def write_generic_method_signature(
    f,
    obj,
    obj2: Function,
    first_arg_name: Optional[str] = 'self',
    caller_name=get_obj_instance_name,
) -> None:
    parameters = list(obj2.parameters)
    if first_arg_name is not None:
        # Strip `self` or `cls` argument.
        p0 = parameters[0]
        assert p0.name == first_arg_name, p0  # method
        parameters = parameters[1:]

    if len(parameters) > 1:
        # One line
        params = '\n' + ',\n'.join(f'  {format_parameter(p)}' for p in parameters) + '\n'
    else:
        # Multiline
        params = ', '.join(format_parameter(p) for p in parameters)

    # For methods, make it look like we're calling it on an instance.
    caller_name = caller_name(obj)

    # All together
    if obj2.returns is not None:
        method_signature = f'{caller_name}.{obj2.name}({params}) -> {obj2.returns}'
    else:
        method_signature = f'{caller_name}.{obj2.name}({params})'
    f.write(f'```python\n{method_signature}\n```\n\n')


def write_function_signature(f, obj2: Function):
    # Strip `self` or `cls` argument.
    parameters = list(obj2.parameters)

    if len(parameters) > 1:
        # One line
        params = '\n' + ',\n'.join(f'  {format_parameter(p)}' for p in parameters) + '\n'
    else:
        # Multiline
        params = ', '.join(format_parameter(p) for p in parameters)

    # All together
    if obj2.returns is not None:
        method_signature = f'{obj2.name}({params}) -> {obj2.returns}'
    else:
        method_signature = f'{obj2.name}({params})'
    f.write(f'```python\n{method_signature}\n```\n\n')


def write_method_signature(f, obj: Object, obj2: Function) -> None:
    if obj2.overloads:
        # Assume the full documentation is in the defined function, not the overrides.
        return

    # Filter out decorators that don't matter.
    decs = [d for d in obj2.decorators if str(d.value) not in ['abc.abstractmethod']]

    if len(decs) == 0:
        # No decorators, write a normal method signature.
        return write_generic_method_signature(f, obj, obj2)

    elif len(decs) == 1:
        (d,) = decs

        # It's a property, write a special property signature.
        if str(d.value) == 'property' or str(d.value) == 'cached_property':
            return write_property_method_signature(f, obj, obj2)

        # It's a classmethod, write a special classmethod signature.
        elif str(d.value) == 'classmethod':
            return write_generic_method_signature(
                f, obj, obj2, first_arg_name='cls', caller_name=lambda obj: obj.name
            )

        elif str(d.value) == 'staticmethod':
            return write_generic_method_signature(
                f, obj, obj2, first_arg_name=None, caller_name=lambda obj: obj.name
            )

        else:
            raise ValueError(
                f'{obj.name}.{obj2.name} has decorator {d.value!r}. '
                f'Please upgrade `write_method_signature` in the doc generation code to handle this.'
            )
    else:
        raise ValueError(
            f'{obj.name}.{obj2.name} has multiple decorators. '
            f'Please upgrade `write_method_signature` in the doc generation code to handle this.'
        )
