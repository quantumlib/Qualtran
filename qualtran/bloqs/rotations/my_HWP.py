from hamming_weight_phasing import HammingWeightPhasing, HammingWeightPhasingWithConfigurableAncilla


import numpy as np
from qualtran import Bloq, CompositeBloq, BloqBuilder, Signature, Register
from qualtran import QBit, QInt, QUInt, QAny
from qualtran.drawing import show_bloq, show_call_graph, show_counts_sigma

orig = HammingWeightPhasing(4, np.pi / 2.0)
print(orig)

mine = HammingWeightPhasingWithConfigurableAncilla(4, 2, np.pi / 2.0)
print(mine)


from qualtran.resource_counting.generalizers import ignore_split_join
hamming_weight_phasing_g, hamming_weight_phasing_sigma = mine.call_graph(max_depth=1, generalizer=ignore_split_join)
show_call_graph(hamming_weight_phasing_g)
show_counts_sigma(hamming_weight_phasing_sigma)
