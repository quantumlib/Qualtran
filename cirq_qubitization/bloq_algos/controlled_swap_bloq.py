from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from typing import Dict
from attrs import frozen

from cirq_qubitization.quantum_graph.bloq import Bloq
from cirq_qubitization.quantum_graph.composite_bloq import CompositeBloq, Soquet, SoquetT
from cirq_qubitization import TComplexity
from cirq_qubitization.quantum_graph.fancy_registers import FancyRegister, FancyRegisters

@frozen
class CSWAP(Bloq):
    """A controlled swap between two-qubits

    Args:
        cv1: Whether the control bit is a positive (1) control.
             Alternatively, control bit can be negatively (0) controlled.

    Registers:
     - ctrl: A two-bit control register.
     - x: first bit to swap
     - y: second bit to swap
    """
    cv1: int = 1

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('ctrl', 1, wireshape=(1,)),
                FancyRegister('x', 1, wireshape=(1,),),
                FancyRegister('y', 1, wireshape=(1,),)
            ]
        )

    def t_complexity(self) -> 'TComplexity':
        """The `TComplexity` for this bloq.

        C-swap is decomposed into two CNOT + 1 Toffoli.
        Each Toffoli is 7 T-gates, 8 Cliffords, and 0 rotations 
        """
        num_toffoli = 1
        return TComplexity(t=num_toffoli * 7, 
                           clifford=2 + 8 * num_toffoli, 
                           rotations=0
        )

@frozen
class CnSWAP(Bloq):
    """A controlled n-bit swap between two registers

    Args:
        cv1: Whether the control bit is a positive (1) control.
             Alternatively, control bit can be negatively (0) controlled.
        reg_length: Length of the registers

    Registers:
     - ctrl: A two-bit control register.
     - reg_x: The first register of size n-bits
     - reg_y: The second register of size n-bits
    """

    reg_length: int = None
    cv1: int = 1

    @cached_property
    def registers(self) -> FancyRegisters:
        return FancyRegisters(
            [
                FancyRegister('ctrl', 1, wireshape=(1,)),
                FancyRegister('reg_x', self.reg_length, wireshape=(1,),),
                FancyRegister('reg_y', self.reg_length, wireshape=(1,),)
            ]
        )
    def pretty_name(self) -> str:
        return f'CSWAP'

    def build_composite_bloq(
        self, bb: 'CompositeBloqBuilder', *, cntrl_and_targets: NDArray[Soquet]
    ) -> Dict[str, 'SoquetT']:
        """Decomposes multi-cswap `CnSWAP` in-terms of an `CSWAP`       
        """    
        ctrl = cntrl_and_targets[0],
        reg_x = cntrl_and_targets[1:self.reg_length+1]
        reg_y = cntrl_and_targets[self.reg_length+1:]
        returned_x = []
        returned_y = []
        for n_idx in range(self.reg_length):
            # overwrite control with same control
            ctrl, out_reg_x, out_reg_y  = bb.add(CSWAP(cv1=self.cv1), 
                                                 ctrl=ctrl, 
                                                 x=reg_x[n_idx], 
                                                 y=reg_y[n_idx]
                                                 )
            returned_x.append(out_reg_x)
            returned_y.append(out_reg_y)
        return {
            'ctrl': np.asarray([ctrl]),
            'reg_x': np.asarray(returned_x),
            'reg_y': np.asarray(returned_y),
        }
