from bloqs.factoring.mod_mul import CtrlModMul
from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
from qualtran.serialization.bloq import arg_to_proto, arg_from_proto
import sympy
# sympy.pi
# sympy.E
# sympy.EulerGamma
# sympy.core.numbers.Infinity
# sympy.core.numbers.ImaginaryUnit


def parameter_test():
    const_int = sympy.parse_expr("5")
    symbol = sympy.symbols("x")
    fraction = sympy.parse_expr("1/2")
    const_symbol = sympy.pi
    const_complex = sympy.parse_expr("2j")

    expr = const_int + symbol + fraction + const_symbol + const_complex

    serialized = arg_to_proto(name="test", val=expr)
    expr_clone = arg_from_proto(serialized)['test']
    print("done")

# parameter_test()



k, N, n_x = sympy.symbols('k N n_x')
bloq = CtrlModMul(k=k, mod=N, bitsize=n_x)

serialized = bloqs_to_proto(bloq)
reconstructed = bloqs_from_proto(serialized)
print("Done")
#
