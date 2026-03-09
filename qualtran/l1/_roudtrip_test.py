from pathlib import Path

import networkx as nx

import qualtran as qlt
import qualtran.bloqs.arithmetic
import qualtran.bloqs.basic_gates
import qualtran.bloqs.chemistry.hubbard_model.qubitization
import qualtran.bloqs.mcmt
import qualtran.dtype as qdt
import qualtran.l1


def check_bloq_keys(original, loaded) -> list[str]:
    problems = []
    ref = set(original.keys())
    tst = set(loaded.keys())
    if ref != tst:
        if len(ref - tst) > 0:
            for x in ref - tst:
                problems.append(f'BLOQ_KEYS: Missing {x}')
        else:
            for x in tst - ref:
                problems.append(f'BLOQ_KEYS: Superfluous {x}')

    return problems


def check_signatures(original, loaded):
    problems = []
    for bloq_key in original.keys():
        ref = original[bloq_key]
        tst = loaded[bloq_key]

        if ref.bloq.signature != tst.signature:
            problems.append(f"SIGNATURE: FAIL: {bloq_key}")
    return problems


def check_bloq_object_roundtrip(original, loaded) -> list[str]:
    problems = []
    for bloq_key in original.keys():
        ref = original[bloq_key]
        tst = loaded[bloq_key]

        if isinstance(tst, qlt.CompositeBloq):

            if ref.bloq != tst.decomposed_from:
                problems.append(f"BOBJ: FAIL {bloq_key}")
        else:
            if ref.bloq != tst:
                problems.append(f'BOBJ EXTERN: FAIL {bloq_key}')

    return problems


def check_soquet_graph_isomorphism(original, loaded) -> list[str]:
    problems = []
    for bloq_key in original.keys():
        ref = original[bloq_key]
        tst = loaded[bloq_key]

        try:
            ref = ref.bloq.decompose_bloq()
        except qlt.DecomposeTypeError:
            continue
        except qlt.DecomposeNotImplementedError:
            continue

        ref_sg = nx.DiGraph()
        ref_sg.add_nodes_from((soq, {'soq': soq}) for soq in ref.all_soquets)
        ref_sg.add_edges_from((cxn.left, cxn.right) for cxn in ref.connections)

        blq_sg = nx.DiGraph()
        blq_sg.add_nodes_from((soq, {'soq': soq}) for soq in tst.all_soquets)
        blq_sg.add_edges_from((cxn.left, cxn.right) for cxn in tst.connections)

        def node_match(n1, n2):
            s1 = n1['soq']
            s2 = n2['soq']
            return s1.reg == s2.reg

        if nx.is_isomorphic(blq_sg, ref_sg, node_match=node_match):
            pass
        else:
            problems.append(f'SOQUET GRAPH ISOMORPHISM: FAIL {bloq_key}')
    return problems


def check_roundtrip(bloq_objectstring: str, bloq: qlt.Bloq):

    loaded_bloq = qualtran.l1.load_bloq(bloq_objectstring)
    assert loaded_bloq == bloq

    # Manually use L1ModuleBuilder so we have a record of all the true subbloqs
    # we're serializing.
    l1mb = qualtran.l1.L1ModuleBuilder()
    root_bloq_key = l1mb.add_bloqs(root=bloq, extern_only_from=False)
    original_bloqs = {qdef.qdef.bloq_key: qdef for qdef in l1mb.qdefs}

    # Get the textual representation
    l1_code = qualtran.l1.L1ASTPrinter().visit(l1mb.finalize())

    # Re-parse the textual representation
    l1_module = qualtran.l1.load_module(l1_code)

    problems = []
    problems += check_bloq_keys(original_bloqs, l1_module)
    problems += check_signatures(original_bloqs, l1_module)
    problems += check_bloq_object_roundtrip(original_bloqs, l1_module)
    problems += check_soquet_graph_isomorphism(original_bloqs, l1_module)
    assert problems == []


def test_cswap_roundtrip():
    check_roundtrip(
        'qualtran.bloqs.basic_gates.CSwap(bitsize=5)', qualtran.bloqs.basic_gates.CSwap(bitsize=5)
    )


def test_negate_roundtrip():
    check_roundtrip(
        'qualtran.bloqs.arithmetic.Negate(QInt(8))', qualtran.bloqs.arithmetic.Negate(qdt.QInt(8))
    )


def test_multiand_roundtrip():
    check_roundtrip(
        'qualtran.bloqs.mcmt.MultiAnd(cvs=(1,1,0,1))',
        qualtran.bloqs.mcmt.MultiAnd(cvs=(1, 1, 0, 1)),
    )


def test_select_hubbard_roundtrip():
    check_roundtrip(
        'qualtran.bloqs.chemistry.hubbard_model.qubitization.SelectHubbard(x_dim=5, y_dim=5)',
        qualtran.bloqs.chemistry.hubbard_model.qubitization.SelectHubbard(5, 5),
    )
