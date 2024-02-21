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
from typing import List, Tuple, Union

import numpy as np
import pyqir
from pyqir import BasicBlock, Builder, Context, Function, Linkage
from pyqir._native import Function, FunctionType, Type

import qualtran
from qualtran import Soquet
from qualtran.bloqs.basic_gates.rotation import CZPowGate

PYQIR_OP_MAP = {
    # Single-Qubit Clifford Gates
    'Hadamard': pyqir._native.h,
    'cirq.H': pyqir._native.h,
    'XGate': pyqir._native.x,
    'YGate': pyqir._native.y,
    'ZGate': pyqir._native.z,
    # Single-Qubit Rotation Gates
    'cirq.Rx': pyqir._native.rx,
    'cirq.Ry': pyqir._native.ry,
    'cirq.Rz': pyqir._native.rz,
    # Single-Qubit Non-Clifford Gates
    'T': pyqir._native.t,
    # Two-Qubit Gates
    'TwoBitSwap': pyqir._native.swap,
    'CNOT': pyqir._native.cx,
    'cirq.CNOT': pyqir._native.cx,
    # Three-Qubit Gates
    'Toffoli': pyqir._native.ccx,
}


def get_num_qubits_for_bloq(bloq: qualtran.Bloq) -> int:
    """
    Get the number of qubits used in the given Bloq.

    Args:
        bloq: Bloq to get number of qubits from.

    Returns:
        The number of qubits used in the given Bloq
    """
    num_qubits = 0
    for register in bloq.signature.lefts():
        shape = register.shape[0] if len(register.shape) != 0 else 1
        num_qubits += register.bitsize * shape
    return num_qubits


def create_func_for_bloq(
    bloq: qualtran.Bloq, name: str, qubit_type: Type, void_type, mod: pyqir.Module
) -> Function:
    """
    Create a QIR function for the given Bloq.

    Args:
        bloq: Bloq to create the QIR function for.
        name: Name of the QIR function.
        qubit_type: Qubit type for the QIR function.
        void_type: Void type for the QIR function.
        mod: Module to add the QIR function to.
    Returns:
        The QIR function for the given Bloq.
    """
    num_qubits = get_num_qubits_for_bloq(bloq)
    return Function(FunctionType(void_type, [qubit_type] * num_qubits), Linkage.EXTERNAL, name, mod)


def create_ir_map(bloq: qualtran.Bloq) -> dict:
    """
    Creates a dictionary which maps the IR of qubits to an index.

    Given that we have a QIR function with a list of parameters, params,
    and len(params) = m:
    The IR map of a Qualtran Bloq register named 'x' with n qubits is a map
    with entries ('x', i): j where i goes from 0 to n - 1 and 0<=j<m.
    The IR map of the entire Qualtran Bloq is the union of the IR of each register in the Bloq.
    Note that for all j in {0,1,..., m - 1}, there should exist a key such that IR[key] = j.

    This function creates such a map.

    Args:
        bloq: Bloq to create the IR map for.

    Returns:
        Dictionary which maps the IR of qubits to an index.
    """
    param_counter = 0
    ir_map = {}
    # Loop through all registers in signature
    for register in bloq.signature.lefts():
        shape = (
            register.shape[0] if len(register.shape) != 0 else 1
        )  # get the shape as an int (we are asumming its 1d for simplicity)
        # map the (register_name, index_in_register) to the overall index
        ir_map.update(
            {
                (register.name, i * register.bitsize + j): param_counter + i * register.bitsize + j
                for i in range(shape)
                for j in range(register.bitsize)
            }
        )
        param_counter += register.bitsize * shape
    return ir_map


def find_ir_for_index(i: int, ir_map: dict) -> Tuple[str, int]:
    """
    Given an index i, find the IR for that index.

    Args:
        i: Index to find the IR for.
        ir_map: The IR map to use.

    Returns:
        The IR for the given index.
    """
    for key, val in ir_map.items():
        if i == val:
            return key
    raise Exception(f"Error in find_ir_for_index: No IR for index {i}")


def cz_power(angle: float, params: List[pyqir._native.Constant], builder, qubit_alloc):
    """
    Controlled-Z power gate implementation in QIR


    Args:
        angle: The angle of the rotation.
        params: The parameters for the gate.
        builder: The QIR builder to use.
        qubit_alloc: The QIR function to allocate qubits.
    """
    ancilla = builder.call(qubit_alloc, [])
    pyqir._native.ccx(builder, *params, ancilla)
    pyqir._native.rz(builder, angle, params[1])
    pyqir._native.ccx(builder, *params, ancilla)


def bloq_decomposes(bloq: qualtran.Bloq) -> bool:
    """
    Check if a bloq decomposes into sub-bloqs.

    Args:
        bloq: The bloq to check.

    Returns:
        True if the bloq decomposes, False otherwise.
    """
    try:
        bloq.decompose_bloq()
        return True
    except:
        return False


def map_soquet_to_param_indices(
    soq: Union[Soquet, np.ndarray], soq_map: dict, ir_map: dict
) -> List[int]:
    """
    Map a soquet to the corresponding list of parameter indices.

    Args:
        soq: The soquet to map.
        soq_map: The soquet map to use.
        ir_map: The IR map to use.

    Returns:
        List of parameter indices corresponding to the soquet.
    """
    if isinstance(soq, np.ndarray):
        # renaming to soqs for clarity
        soqs = soq
        return [
            el for soq in soqs for el in map_single_soquet_to_param_indices(soq, soq_map, ir_map)
        ]
    return map_single_soquet_to_param_indices(soq, soq_map, ir_map)


def map_single_soquet_to_param_indices(soquet, soq_map, ir_map) -> List[int]:
    """
    Map a single soquet to the corresponding list of parameter indices.

    Args:
        soquet: The soquet to map.
        soq_map: The soquet map to use.
        ir_map: The IR map to use.

    Returns:
        List of parameter indices corresponding to the soquet.
    """
    if soquet in soq_map:
        irs = soq_map[soquet]
        return [ir_map[ir] for ir in irs]
    irs = irs_from_soquet(soquet)
    return [ir_map[ir] for ir in irs]


def irs_from_soquet(soq) -> List[Tuple[str, int]]:
    """
    Get the IRs from a soquet.

    Args:
        soq: The soquet to get the IRs from.

    Returns:
        List of IRs from the soquet.
    """
    reg_name = soq.reg.name
    starting_index = soq.idx[0] if len(soq.idx) != 0 else 0
    return [(reg_name, starting_index * soq.reg.bitsize + i) for i in range(soq.reg.bitsize)]


def compile_bloq(
    bloq: qualtran.Bloq,
    qubit_type: Type,
    void_type: Type,
    module: pyqir.Module,
    context: pyqir.Context,
    builder: Builder,
    qubit_alloc: Function,
    func_dict=dict(),
) -> Tuple[Function, dict]:
    """
    Compile a Bloq into a QIR function.

    This function, at a high level, takes in a Bloq, creates a corresponding QIR function,
    and then recursively goes through its sub-bloqs until it reaches a base case in which the bloq
    can't be decomposed any further. The intent is for these leaf bloqs to be in the PYQIR_OP_MAP (defined above)
    in which case the corresponding QIR operation is called. In the case where the leaf bloq is not in PYQIR_OP_MAP,
    the leaf bloq is mapped to a QIR function without an implementation.

    Args:
        bloq: Bloq to compile.
        qubit_type: Qubit type for the QIR function.
        void_type: Void type for the QIR function.
        module: Module to add the QIR function to.
        context: Context to use.
        builder: Builder to use.
        qubit_alloc: Function to allocate qubits.
        func_dict: Dictionary to store the functions already created.

    Returns:
        Tuple containing the QIR function for the given Bloq and the IR map for the Bloq.
    """
    func_name = f"{bloq.pretty_name()}_{get_num_qubits_for_bloq(bloq)}"
    if func_name in func_dict:
        return func_dict[func_name]
    bloq_func = create_func_for_bloq(bloq, func_name, qubit_type, void_type, module)
    ir_map = create_ir_map(bloq)
    soq_map = {}

    if not bloq_decomposes(bloq):
        return bloq_func, ir_map

    # Create a block to insert instructions in
    basic_block = BasicBlock(context, "block1", bloq_func)

    # iterate through the DAG of sub bloqs
    for sub_bloq in bloq.decompose_bloq().iter_bloqsoqs():
        bloq_instance, inputs, outputs = sub_bloq
        if bloq_instance.bloq.short_name() == 'Split':
            reg_name = inputs['reg'].reg.name
            for i in range(len(outputs[0])):
                ir = soq_map[inputs['reg']][i] if inputs['reg'] in soq_map else (reg_name, i)
                soq_map[outputs[0][i]] = [ir]
            continue
        if bloq_instance.bloq.short_name() == 'Join':
            irs = [el for soq in inputs['reg'] for el in soq_map[soq]]
            soq_map[outputs[0]] = irs
            continue

        if type(bloq_instance.bloq) == CZPowGate:
            builder.insert_at_end(basic_block)
            param_indexes = [
                param
                for soquet in inputs.values()
                for param in map_soquet_to_param_indices(soquet, soq_map, ir_map)
            ]
            params = [bloq_func.params[i] for i in param_indexes]
            cz_power(bloq_instance.bloq.exponent, params, builder, qubit_alloc)
        elif bloq_instance.bloq.pretty_name() in PYQIR_OP_MAP:
            builder.insert_at_end(basic_block)
            param_indexes = [
                param
                for soquet in inputs.values()
                for param in map_soquet_to_param_indices(soquet, soq_map, ir_map)
            ]
            params = [bloq_func.params[i] for i in param_indexes]
            PYQIR_OP_MAP[bloq_instance.bloq.pretty_name()](builder, *params)
        else:
            param_indexes = [None for _ in range(get_num_qubits_for_bloq(bloq_instance.bloq))]

            sub_bloq_func, sub_bloq_ir_map = compile_bloq(
                bloq_instance.bloq,
                qubit_type,
                void_type,
                module,
                context,
                builder,
                qubit_alloc,
                func_dict,
            )
            sub_func_name = (
                f"{bloq_instance.bloq.pretty_name()}_{get_num_qubits_for_bloq(bloq_instance.bloq)}"
            )
            func_dict[sub_func_name] = sub_bloq_func, sub_bloq_ir_map
            for key in inputs.keys():
                caller_param_indices_for_key = map_soquet_to_param_indices(
                    inputs[key], soq_map, ir_map
                )
                callee_param_indices_for_key = [
                    sub_bloq_ir_map[(reg_name, i)]
                    for (reg_name, i) in sub_bloq_ir_map.keys()
                    if reg_name == key
                ]
                for param_index, qubit_param_index in list(
                    zip(callee_param_indices_for_key, caller_param_indices_for_key)
                ):
                    param_indexes[param_index] = qubit_param_index

            builder.insert_at_end(basic_block)
            params = [bloq_func.params[i] for i in param_indexes]
            builder.call(sub_bloq_func, params)

        # map soquets to ir
        param_index = 0
        for soq in outputs:
            if not isinstance(soq, np.ndarray):
                soq = np.array([soq])
            for single_soq in soq:
                irs = []
                for i in range(single_soq.reg.bitsize):
                    irs.append(find_ir_for_index(param_indexes[param_index], ir_map))
                    param_index += 1
                soq_map[single_soq] = irs
    builder.ret(None)

    return bloq_func, ir_map


def bloq_to_qir(bloq: qualtran.Bloq) -> pyqir.Module:
    """
    Convert a Bloq to QIR.

    Args:
        bloq: Bloq to convert to QIR.

    Returns:
        The QIR module for the given Bloq.
    """
    context = Context()
    mod = pyqir.qir_module(
        context,
        bloq.pretty_name(),
        qir_major_version=1,
        qir_minor_version=0,
        dynamic_qubit_management=True,
    )
    builder = Builder(context)
    qubit_type = pyqir.qubit_type(context)
    void_type = pyqir.Type.void(context)
    entry = pyqir.entry_point(mod, "main", get_num_qubits_for_bloq(bloq), 0)
    qubit_allocate = Function(
        pyqir.FunctionType(qubit_type, []), Linkage.EXTERNAL, "__quantum__rt__qubit_allocate", mod
    )
    entry_block = BasicBlock(context, "entry", entry)
    builder.insert_at_end(entry_block)
    qubits = [pyqir.qubit(context, n) for n in range(get_num_qubits_for_bloq(bloq))]
    bloq_func, _ = compile_bloq(bloq, qubit_type, void_type, mod, context, builder, qubit_allocate)
    builder.insert_at_end(entry_block)
    builder.call(bloq_func, qubits)
    builder.ret(None)
    return mod
