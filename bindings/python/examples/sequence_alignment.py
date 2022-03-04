"""
This script shows how to use GTN to compute the Needleman-Wunsch global
alignment and Smith-Waterman local alignment between two sequences.

The script supports both linear gap scores and affine gap scores (i.e. a gap
open and gap extension penalty which are different).

The example comes from Chapter 2 of
"Biological Sequence Analysis", Durbin et al.
"""
import gtn
import json


RESIDUE_MAP = {r: i for i, r in enumerate("ARNDCQEGHILKMFPSTWYV")}


def load_blosum():
    with open("blosum.json", 'r') as fid:
        return json.load(fid)


def make_score_graph(gap_open=-8, gap_add=-8):
    blosum = load_blosum()
    g = gtn.Graph()
    g.add_node(True, True)
    affine = (gap_open != gap_add)
    if affine:
        g.add_node(False, True)
        g.add_node(False, True)
    for k, v in blosum.items():
        r1, r2 = k
        g.add_arc(0, 0, RESIDUE_MAP[r1], RESIDUE_MAP[r2], v)
        if affine:
            g.add_arc(1, 0, RESIDUE_MAP[r1], RESIDUE_MAP[r2], v)
            g.add_arc(2, 0, RESIDUE_MAP[r1], RESIDUE_MAP[r2], v)
    if affine:
        for r in RESIDUE_MAP.values():
            g.add_arc(0, 1, r, gtn.epsilon, gap_open)
            g.add_arc(1, 1, r, gtn.epsilon, gap_add)
            g.add_arc(0, 2, gtn.epsilon, r, gap_open)
            g.add_arc(2, 2, gtn.epsilon, r, gap_add)
    else:
        for r in RESIDUE_MAP.values():
            g.add_arc(0, 0, r, gtn.epsilon, gap_open)
            g.add_arc(0, 0, gtn.epsilon, r, gap_open)
    return g


def make_seq_graph(seq, alg="nw"):
    g = gtn.Graph()
    start = (alg == "sw")
    accept = (alg == "sw")
    g.add_node(start=True, accept=accept)
    for e, s in enumerate(seq):
        g.add_node(
            start=start,
            accept=accept or (e == len(seq) - 1))
        g.add_arc(e, e + 1, RESIDUE_MAP[s])
    return g


def compute_path_and_score(seq_a, seq_b, alg):
    score_graph = make_score_graph()
    seq_graph_a = make_seq_graph(seq_a, alg)
    seq_graph_b = make_seq_graph(seq_b, alg)
    alis_graph = gtn.compose(gtn.compose(seq_graph_a, score_graph), seq_graph_b)
    ali_path = gtn.viterbi_path(alis_graph)
    ali_score = gtn.viterbi_score(alis_graph)
    return ali_path, ali_score


def align_and_print(seq_a, seq_b, alg):
    inv_labels = {v: k for k, v in RESIDUE_MAP.items()}
    inv_labels[gtn.epsilon] = "*"
    ali_path, ali_score = compute_path_and_score(seq_a, seq_b, alg)
    in_seq = ali_path.labels_to_list()
    out_seq = ali_path.labels_to_list(False)
    print(f"Alignment score: {ali_score.item():.2f}")
    print("".join(inv_labels[r] for r in in_seq))
    print("".join(inv_labels[r] for r in out_seq))


if __name__ == "__main__":
    seq_a = "HEAGAWGHEE"
    seq_b = "PAWHEAE"
    print("="*20)
    print("Needleman-Wunsch global alignment")
    align_and_print(seq_a, seq_b, alg="nw")
    print("="*20)
    print("Smith-Waterman local alignment")
    align_and_print(seq_a, seq_b, alg="sw")
