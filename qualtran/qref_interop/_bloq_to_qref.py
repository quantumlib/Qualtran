import networkx as nx

from functools import singledispatch
from typing import Any, Iterable, Optional, Union

import sympy
from qref.schema_v1 import PortV1, RoutineV1, SchemaV1

from qualtran import Bloq, BloqInstance, CompositeBloq
from qualtran import Connection as QualtranConnection
from qualtran import Register, Side, Soquet
from qualtran.cirq_interop import CirqGateAsBloq


@singledispatch
def _bloq_type(bloq: Bloq) -> str:
    """Determine type of Bloq.

    Output of this function is used for populating the `type` field of RoutineV1
    build from imported Bloqs. By default, we use name of the type of the Bloq.
    """
    return type(bloq).__name__


@_bloq_type.register
def _cirq_gate_bloq_type(bloq: CirqGateAsBloq) -> str:
    """Determine type of Bloq constructed from Cirq gate.

    Without this variant of _bloq_type, type of all instances of CirqGateAsBloq
    are set to "CirqGateAsBloq" which is not helpful, because all gates would get the
    same type. Instead of this default behaviour, we extract textual representation
    of the Cirq's gate.
    """
    return str(bloq.gate)


def _is_symbol_or_int(expression):
    try:
        int(expression)
        return True
    except (TypeError, ValueError):
        return expression.isidentifier()


def _extract_common_bloq_attributes(bloq: Bloq, name: Optional[str] = None) -> dict[str, Any]:
    """Extract common bloq attributes such as name, type and ports.

    There are several Bloq classes, however, they all share common set of atributes.
    This function is used to extract them, so that we don't have to duplicate logic
    for each Bloq subtype.

    Args:
        bloq: Bloq for which information should be extracted.
        name: Optional override for name. Setting it to None will cause some reasonable
            name (depending on bloq type) to be inferred. The typical use of non-trivial
            name is when a better name is known from the parent Bloq.

    Returns:
        A dictionary that can be unpacked into arguments of RoutineV1 initializer.
    """
    ports = [port for reg in bloq.signature for port in _ports_from_register(reg)]
    local_variables = {}
    for port in ports:
        if not _is_symbol_or_int(str(port.size)):
            local_variable_name = f"{port.name}_size"
            local_variables[local_variable_name] = port.size
            port.size = local_variable_name

    if name is None:
        name = bloq.__class__.__name__
        try:
            if bloq.uncompute:
                name += "_uncompute"
        except AttributeError:
            pass
        try:
            if bloq.is_adjoint:
                name += "_adjoint"
        except AttributeError:
            pass
    input_params = _extract_input_params(bloq, ports, local_variables)

    attributes = {
        "name": name,
        "type": _bloq_type(bloq),
        "ports": ports,
        "resources": _import_resources(bloq),
        "input_params": input_params,
    }
    if len(local_variables) > 0:
        attributes["local_variables"] = local_variables
    return attributes


def _bloq_instance_name(instance: BloqInstance) -> str:
    """Infer unique (but readable) name for a BloqInstance.

    Child Bloqs in CompositeBloq (and some other places) are stored as BloqInstances,
    which combine a Bloq with a unique ID. When converting such BloqInstance to Bartiq
    RoutineV1, the ID has to be incorporated into the name, because otherwise one could
    get several siblings having the same name.
    """
    return f"{_bloq_type(instance.bloq)}_{instance.i}"


def bloq_to_qref(obj) -> SchemaV1:
    """Converts Bloq to QREF SchemaV1 object."""
    return SchemaV1(version="v1", program=bloq_to_routine(obj))


@singledispatch
def bloq_to_routine(obj, name: Optional[str] = None):
    """Import object from Qualtran by converting it into corresponding QREF RoutineV1 object.

    Args:
        obj: object to be imported. Can be either Bloq or BloqInstance.
        name: optional name override. This can be useful e.g. if you are converting
            CompositeBloq (which do not store meaningful names in Qualtran) and
            know some good name for it.

    Return:
        A Bartiq object corresponding to the source Qualtran object. For both Bloqs
        and BloqInstances the returned object is of type RoutineV1.

    """
    raise NotImplementedError(f"Cannot import object {obj} of type {type(obj)}.")


@bloq_to_routine.register
def _composite_bloq_to_routine(bloq: CompositeBloq, name: Optional[str] = None) -> RoutineV1:
    """Import CompositeBloq from Qualtran.

    See `import_from_qualtran` for more info.
    """
    return RoutineV1(
        **_extract_common_bloq_attributes(bloq, name),
        children=[bloq_to_routine(instance) for instance in bloq.bloq_instances],
        connections=[_import_connection(c) for c in bloq.connections],
    )


@bloq_to_routine.register
def _bloq_to_routine(bloq: Bloq, name: Optional[str] = None) -> RoutineV1:
    """Import Bloq (other than CompositeBloq) from Qualtran.

    See `import_from_qualtran` for moe info.
    """
    return RoutineV1(**_extract_common_bloq_attributes(bloq, name))


@bloq_to_routine.register
def _bloq_instance_to_routine(instance: BloqInstance) -> RoutineV1:
    """Import Bloq (other than CompositeBloq) from Qualtran.

    When importing BloqInstance we derive name from Bloq's default name and
    instance ID to prevent duplication of names between siblings.

    See `import_from_qualtran` and `_bloq_instance_name` for more info.
    """
    return bloq_to_routine(instance.bloq, name=_bloq_instance_name(instance))


def _names_and_dir_from_register(reg: Register) -> Iterable[tuple[str, str]]:
    """Yield names and directions of Bartiq Ports corresponding to QualtranRegister.

    For LEFT/RIGHT registers we yield one pair of name and direction corresponding
    of resp. output and input port. For THRU registers we yield both such pairs,
    which effectively splits THRU registers into two ports.
    """
    if reg.side != Side.LEFT:
        yield (f"out_{reg.name}", "output")
    if reg.side != Side.RIGHT:
        yield (f"in_{reg.name}", "input")


def _expand_name_if_needed(reg_name, shape) -> Iterable[str]:
    """Given a register name, expand it into sequence of names if it has nontrivial shape.

    Examples:
        "reg", () -> "reg"
        "reg", (3,) -> "reg_0", "reg_1", "reg_2"
    """
    return (reg_name,) if shape == () else (f"{reg_name}_{i}" for i in range(shape[0]))


def _ports_from_register(reg: Register) -> Iterable[PortV1]:
    """Given a Qualtran register, return iterable of corresponding Bartiq Ports.

    Intuitively, one would expect a one to one correspondence between ports and registers.
    However:
        - We currently don't support bidirectional ports. Hence, THRU registers have to be
          split into pairs of input/output ports
        - We currently don't support composite ports. Hence composite registers (i.e. ones
          with a nontrivial shape) have to be split into multiple ports.

    Args:
        reg: Register to be converted into ports.

    Raises:
       NotImplementedError: if `reg` is a compound register with more than one-dimension.
       Currently we don't support such scenario.
    """
    if len(reg.shape) > 1:
        raise NotImplementedError(
            "Registers with two or more dimensional shape are not yet supported. "
            f"The error was caused by register {reg}."
        )

    return sorted(
        [
            # Observe two loops:
            # - first one splits (if needed) any THRU register into two ports. It also takes care of
            #   correct naming based on port directions.
            # - second one expands composite register (which have no counterpart in Bartiq) into
            #   required number of single ports.
            PortV1(
                name=expanded_name,
                direction=direction,
                size=reg.bitsize if isinstance(reg.bitsize, int) else str(reg.bitsize),
            )
            for flat_name, direction in _names_and_dir_from_register(reg)
            for expanded_name in _expand_name_if_needed(flat_name, reg.shape)
        ],
        key=lambda p: p.name,
    )


def _opposite(direction: str) -> str:
    return "out" if direction == "in" else "in"


def _relative_port_name(soquet: Soquet, direction) -> str:
    """Given a Soquet and direction, determine the relative name of corresponding Bartiq Port.

    The relative name is always computed wrt. the parent RoutineV1.
    This function correctly recognizes the fact, that in any connection, the input parent
    ports serve as outputs for the connection (and vice versa).
    """
    if len(soquet.idx) > 1:
        raise NotImplementedError(
            "Soquets referencing more than one index in composite register are not yet supported. "
            f"The error was caused by the following soquet: {soquet}."
        )
    # We add another suffix iff soquet references idx in composite register
    suffix = f"_{soquet.idx[0]}" if soquet.idx else ""
    return (
        # If soquet references BlogqInstance, the corresponding object in Bartiq
        # references child - construct dotted relative name.
        # Otherwise, soquet references the parent port, and so for the output direction,
        # the port is in_parent_port, which is why we include _opposte here.
        f"{_bloq_instance_name(soquet.binst)}.{direction}_{soquet.reg.name}{suffix}"
        if isinstance(soquet.binst, BloqInstance)
        else f"{_opposite(direction)}_{soquet.reg.name}{suffix}"
    )


def _import_connection(connection: QualtranConnection) -> dict[str, Any]:
    """Import connection from Qualtran."""
    return {
        "source": _relative_port_name(connection.left, "out"),
        "target": _relative_port_name(connection.right, "in"),
    }


def _import_resources(bloq: Bloq) -> list[dict[str, Any]]:
    """Import resources from Bloq's t_complexity method."""
    t_complexity = bloq.t_complexity()
    resource_names = ["t", "clifford", "rotations"]
    resources = []
    for name in resource_names:
        resources.append(
            {
                "name": name,
                "value": _ensure_primitive_type(getattr(t_complexity, name)),
                "type": "additive",
            }
        )
    return resources


def _ensure_primitive_type(value: Any) -> Union[int, float, str, None]:
    """Ensure given value is of primitive type (e.g. is not a sympy expression)."""
    return value if value is None or isinstance(value, (int, float, str)) else str(value)


def _extract_input_params(bloq: Bloq, ports: list[PortV1], local_variables) -> list[str]:
    """Extracts input_params from bloq's t_complexity and port sizes."""
    params_from_t_complexity = _extract_input_params_from_t_complexity(bloq)
    params_from_ports = _extract_input_params_from_ports(ports)
    params = list(set(params_from_t_complexity + params_from_ports) - set(local_variables))
    return sorted(params)


def _extract_input_params_from_t_complexity(bloq: Bloq) -> list[str]:
    input_params = set()
    t_complexity = bloq.t_complexity()
    resource_names = ["t", "clifford", "rotations"]
    for name in resource_names:
        try:
            input_params.update(getattr(t_complexity, name).free_symbols)
        except AttributeError:
            pass
    return [str(symbol) for symbol in input_params]


def _extract_input_params_from_ports(ports) -> list[str]:
    input_params = set()
    for port in ports:
        input_params = input_params | sympy.sympify(port.size).free_symbols

    return [str(param) for param in input_params]