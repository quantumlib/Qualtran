from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
import sympy
from qualtran.bloqs.factoring.mod_mul import CtrlModMul

k, N, n_x = sympy.symbols('k N n_x')
bloq = CtrlModMul(k=k, mod=N, bitsize=n_x)

proto = bloqs_to_proto(bloq)
clone = bloqs_from_proto(proto)
print("here")