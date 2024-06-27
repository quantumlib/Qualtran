from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
from qualtran.qref_interop import bloq_to_qref
from qualtran.drawing.graphviz import PrettyGraphDrawer
from qref.experimental.rendering import to_graphviz
from bartiq.integrations.qref import qref_to_bartiq
from bartiq import compile_routine, evaluate
from qualtran.bloqs.data_loading.qrom import QROM

import yaml
import sympy

bloq = StatePreparationAliasSampling.from_lcu_probs(
    [0.25, 0.5, 0.25], probability_epsilon=0.05
)  # .decompose_bloq()

N, M, b1, b2, c = sympy.symbols("N M b1 b2 c")
# N = 10
# M = 15
# b1 = 5
# b2 = 3
# c = 10
# bloq = QROM.build_from_bitsize((N, M), (b1, b2), num_controls=c)
# bloq = QROM.build_from_bitsize((N, M), (b1, b2), num_controls=c).decompose_bloq()


bloq_image = PrettyGraphDrawer(bloq).get_svg_bytes()

with open("bloq.svg", "wb") as f:
    f.write(bloq_image)

qref_definition = bloq_to_qref(bloq)

with open("qref.yaml", "w") as f:
    yaml.safe_dump(qref_definition.model_dump(exclude_unset=True), f, default_flow_style=None)

graphviz_object = to_graphviz(qref_definition)
graphviz_object.render("qualtran_to_qref")


routine = qref_to_bartiq(qref_definition)
print("Initial resources:")
print(routine.resources)
# routine.resources = {}
# print("Empty resources:")
# print(routine.resources)

# breakpoint()

compiled_routine = compile_routine(routine)
print("Compiled resources:")
print(compiled_routine.resources)

# assignments = ["c=20"]
# evaluated_routine = evaluate(compiled_routine, assignments)
# print("Evaluated resources:")
# print(evaluated_routine.resources)

breakpoint()
