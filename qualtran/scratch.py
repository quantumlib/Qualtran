from bloqs.factoring.mod_mul import CtrlModMul
from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
import sympy

# k, N, n_x = sympy.symbols('k N n_x')
# bloq = CtrlModMul(k=k, mod=N, bitsize=n_x)
#
# serialized = bloqs_to_proto(bloq)
# reconstructed = bloqs_from_proto(serialized)
# print("Done")
# #

# Put this into dev_tools
from dev_tools.qualtran_dev_tools.bloq_report_card import get_bloq_report_card, show_bloq_report_card, record_for_bloq_example, check_bloq_example_serialize, get_bloq_examples
from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto


wanted = [
    # "add_diff_size_regs",
    # "add_large",
    "add_small",
    "add_oop_large",
    "add_oop_small",
    "int_effect",
    "int_state",
    "df_one_body",
    "prepare_t",
    "prep_nu_proj",
    "prep_sparse",
    "sel_sparse",
    "thc_prep",
    "qrom_multi_data",
    "qrom_multi_dim",
    "qrom_small",
    "lprs_interim_prep",
    "lp_resource_state_small",
    "qubitization_qpe_hubbard_model_small",
    "textbook_qpe_small",
    "approximate_qft_from_epsilon",
    "approximate_qft_small",
    "two_bit_ffft",
    "reflection",
    "qvr_zpow"
]
bexamples = get_bloq_examples()
my_examples = [example for example in bexamples if example.name in wanted]
for example in my_examples:
    print(example.name, check_bloq_example_serialize(example))


    bloq_lib = bloqs_to_proto(example)
    bloq_roundtrip = bloqs_from_proto(bloq_lib)[0]
print("Done")