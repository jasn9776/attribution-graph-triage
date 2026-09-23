"""ct_utils - frozen metric pipeline for attribution-graph-triage.

Every decision below was settled on Day 1 and is documented in day1_config.json.
Day 3 imports this module; do not edit it without adding a deviations entry.

Settled:
  - Node types recovered from graph STRUCTURE, not index conventions: error and
    embedding nodes have zero in-degree; logits have zero out-degree.
  - Errors and embeddings sit in ONE contiguous zero-in-degree run, errors first
    (CHOICE = errors_first, confirmed against the library's own replacement score).
  - The error block is LAYER-MAJOR. Confirmed on 22-09 by predicting all 30 structural
    zeros from the architecture and matching exactly: BOS at every layer, plus the last
    layer at every non-final position. Position-major does not match.
  - Error at the BOS position is structurally zero. The Day 1 claim that FINAL-position
    error is also zero was an artefact of the logit-ordering bug and is withdrawn
    (D3.10): final-position error is typically the LARGEST. Position summaries use
    positions 1..n-1.
  - A graph that saturates max_feature_nodes is REJECTED, not recorded: saturation
    injects spurious error mass and would fake a category effect.
"""
import numpy as np
import scipy.sparse as sp
import torch


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def contiguous_blocks(idx):
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def find_nodes(A0, n_tok, n_layers, n_log):
    """Return (err_idx, emb_idx, logit_idx). A0 is indexed [target, source]."""
    nz = np.abs(A0) > 0
    zin = np.where(nz.sum(axis=1) == 0)[0]
    zout = np.where(nz.sum(axis=0) == 0)[0]

    run = max(contiguous_blocks(zin), key=lambda b: b[1] - b[0])
    span = run[1] - run[0] + 1
    want = n_layers * n_tok + n_tok
    if span != want:
        raise ValueError(f"zero-in-degree run is {span}, expected {want}")

    ne = n_layers * n_tok
    err = np.arange(run[0], run[0] + ne)
    emb = np.arange(run[0] + ne, run[1] + 1)

    lg = np.setdiff1d(zout, zin)
    if len(lg) != n_log:
        lg = lg[np.argsort(-np.abs(A0[lg]).sum(axis=1))[:n_log]]
    return err, emb, lg


def influence(A_raw, logit_idx, logit_probs, absorb_idx, max_iter=500):
    """Probability-weighted total path strength from each node to the logits.

    Row-normalise incoming edge magnitudes, then push mass backwards from the
    logits one edge per step. The graph is a DAG, so the series terminates
    exactly rather than converging asymptotically. Returns the normalised
    matrix, the influence vector, mass absorbed at `absorb_idx` per path length,
    and the number of steps taken.
    """
    A = np.abs(A_raw)
    rs = A.sum(axis=1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    AT = sp.csr_matrix(A.T)

    v = np.zeros(A.shape[0])
    v[logit_idx] = logit_probs
    infl = np.zeros(A.shape[0])
    by_len = []
    for k in range(1, max_iter + 1):
        v = AT @ v
        infl += v
        by_len.append(v[absorb_idx].sum())
        if v.sum() < 1e-12:
            break
    return A, infl, np.array(by_len), k


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0 or len(x) < 2:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def thirds(v):
    tot = v.sum()
    if tot <= 0:
        return [0.0, 0.0, 0.0]
    return [float(p.sum() / tot) for p in np.array_split(v, 3)]


def pruning_curve(graph, prune_fn, thresholds=(0.95, 0.9, 0.8, 0.7)):
    out = {}
    if prune_fn is None:
        return out
    for t in thresholds:
        try:
            res = prune_fn(graph, node_threshold=t, edge_threshold=t)
            mask = getattr(res, "node_mask", None)
            out[str(t)] = int(mask.sum()) if mask is not None else None
        except Exception as ex:
            out[str(t)] = f"ERR {type(ex).__name__}"
    return out


def graph_metrics(gr, cfg, prune_fn=None):
    """SUPERSEDED (22-09-2026): locates logits structurally and can mis-order them.
    Kept only as a record of the Day 1 implementation. Use graph_metrics_fast."""
    A0 = to_np(gr.adjacency_matrix).astype(np.float64)
    nt = len(to_np(gr.input_tokens))
    nlg = len(to_np(gr.logit_probabilities))
    nl = cfg["n_layers"]
    cap = cfg["attr_kwargs"]["max_feature_nodes"]

    n_feat = A0.shape[0] - (nl * nt + nt + nlg)
    if n_feat >= cap:
        raise ValueError(f"feature cap saturated: {n_feat} >= {cap}")

    err, emb, lg = find_nodes(A0, nt, nl, nlg)

    lw = to_np(gr.logit_probabilities)
    lw = lw / lw.sum()
    A, inf, bl, steps = influence(A0, lg, lw, emb)
    if steps >= 500:
        raise ValueError("influence series did not terminate: orientation is wrong")

    rep = float(inf[emb].sum())
    etot = float(inf[err].sum())
    if abs(rep + etot - 1.0) > cfg["tolerance"]:
        raise ValueError(f"conservation violated: {rep + etot}")

    mask = np.zeros(A.shape[0], bool)
    mask[err] = True
    w = inf * (A.sum(axis=1) > 0)
    comp = float(1 - (w @ A[:, mask].sum(axis=1)) / w.sum())

    grid = inf[err].reshape(nl, nt)          # layer-major, settled Day 1
    by_pos = grid.sum(axis=0)
    by_layer = grid.sum(axis=1)
    interior = by_pos[1:-1] if nt > 2 else by_pos

    return {
        "replacement_score": rep,
        "completeness_score": comp,
        "error_mass_total": etot,
        "mean_path_length": float((np.arange(1, len(bl) + 1) @ bl) / bl.sum()),
        # position summaries on interior positions (BOS and final are structurally zero)
        "err_pos_thirds": thirds(interior),
        "err_pos_argmax_norm": float(np.argmax(interior) / max(len(interior) - 1, 1)),
        "err_pos_gini": gini(interior),
        "err_second_last_pos": float(by_pos[-2]) if nt > 1 else 0.0,
        # layer summaries - always 26 values, comparable across the whole corpus
        "err_layer_thirds": thirds(by_layer),
        "err_layer_argmax": int(np.argmax(by_layer)),
        "err_late_layers": float(by_layer[-6:].sum()),
        # structural-zero diagnostics; should be ~0 on every graph
        "err_bos_pos": float(by_pos[0]),
        "err_final_pos": float(by_pos[-1]),
        # raw profiles, for anything not pre-specified
        "error_by_position": by_pos.tolist(),
        "error_by_layer": by_layer.tolist(),
        "n_nodes": int(A.shape[0]),
        "n_features": int(n_feat),
        "n_edges": int((A0 != 0).sum()),
        "n_tokens": nt,
        "series_steps": int(steps),
        "pruning_nodes": pruning_curve(gr, prune_fn),
    }


# ---------------------------------------------------------------------------
# FAST PATH (added after Day 1 final showed 50-260 s per graph at cap 24576)
#
# graph_metrics above converts the adjacency to dense float64 NumPy on the CPU,
# copying an ~18,000 x 18,000 matrix several times, and calls prune_graph four
# times. graph_metrics_fast does the same computation on the GPU in float32,
# never copies the full matrix off the device, and replaces the library pruning
# curve with a node-influence threshold curve computed from the influence vector
# it already has.
#
# It MUST be validated against graph_metrics on the same graph before use:
# see validate_fast() below.
# ---------------------------------------------------------------------------

def node_count_curve(infl, thresholds=(0.95, 0.9, 0.8, 0.7)):
    """Nodes needed to reach each cumulative share of total influence.

    This is the node-threshold step of the library's prune_graph, without its
    edge pruning or iterative orphan removal. A cheap proxy, reported as such.
    """
    s = np.sort(np.asarray(infl))[::-1]
    tot = s.sum()
    if tot <= 0:
        return {str(t): 0 for t in thresholds}
    c = np.cumsum(s) / tot
    return {str(t): int(np.searchsorted(c, t) + 1) for t in thresholds}


@torch.no_grad()
def graph_metrics_fast(gr, cfg, device="cuda"):
    """GPU float32 metrics, matching circuit_tracer.graph.compute_graph_scores.

    Position summaries exclude BOS only (D3.10). err_final_share is a secondary summary
    recording how much of the position profile the read-out position holds.

    Node layout is taken from the graph, exactly as the library does:
        [features | errors (n_tok * n_layers, layer-major) | tokens | logits]
    with logits the LAST n_logits nodes, in the same order as logit_probabilities.

    An earlier version located logits structurally (zero out-degree). That set also
    contains dead features, and narrowing it by incoming weight REORDERED the logits,
    so each probability was assigned to the wrong node. That was the source of the
    ~0.008 gap from the library. The structural findings are kept as assertions.
    """
    A = gr.adjacency_matrix.to(device=device, dtype=torch.float32, copy=True).abs_()
    N = A.shape[0]
    nt = int(len(to_np(gr.input_tokens)))
    nlg = int(len(to_np(gr.logit_probabilities)))
    nl = cfg["n_layers"]
    cap = cfg["attr_kwargs"]["max_feature_nodes"]

    n_feat = int(len(gr.selected_features))
    if n_feat >= cap:
        raise ValueError(f"feature cap saturated: {n_feat} >= {cap}")
    err_end = n_feat + nl * nt
    tok_end = err_end + nt
    if tok_end + nlg != N:
        raise ValueError(f"layout mismatch: {n_feat}+{nl*nt}+{nt}+{nlg} != {N} nodes")
    err = np.arange(n_feat, err_end)
    emb = np.arange(err_end, tok_end)
    lg = np.arange(N - nlg, N)

    # Structural checks: errors and tokens have no inputs; logits have no outputs.
    nz = A > 0
    in_deg = nz.sum(1).cpu().numpy()
    out_deg = nz.sum(0).cpu().numpy()
    del nz
    if in_deg[n_feat:tok_end].any():
        raise ValueError("an error or token node has incoming edges: layout is wrong")
    if out_deg[lg].any():
        raise ValueError("a logit node has outgoing edges: layout is wrong")

    rs = A.sum(1, keepdim=True)
    A.div_(rs.clamp_min(1e-30))               # rows with no inputs stay zero

    lw = torch.zeros(N, device=device)
    lw[N - nlg:] = torch.as_tensor(to_np(gr.logit_probabilities),
                                   device=device, dtype=torch.float32)
    lw = lw / lw.sum()        # scale cancels in both scores; normalising keeps conservation = 1

    v = lw.clone()
    infl = torch.zeros(N, device=device)
    emb_t = torch.as_tensor(emb, device=device)
    err_t = torch.as_tensor(err, device=device)
    by_len = []
    steps = 0
    for steps in range(1, 501):
        v = v @ A                             # v_new[s] = sum_t v[t] * A[t, s]
        infl += v
        by_len.append(float(v[emb_t].sum()))
        if float(v.sum()) < 1e-12:
            break
    if steps >= 500:
        raise ValueError("influence series did not terminate")

    tok_inf = float(infl[emb_t].sum())
    err_inf = float(infl[err_t].sum())
    if abs(tok_inf + err_inf - 1.0) > cfg["tolerance"]:
        raise ValueError(f"conservation violated: {tok_inf + err_inf}")
    rep = tok_inf / (tok_inf + err_inf)

    # Library completeness: over ALL nodes, weighted by influence PLUS the logit weights.
    non_err = 1 - A[:, err_t].sum(1)
    out_inf = infl + lw
    comp = float((non_err * out_inf).sum() / out_inf.sum())

    inf_np = infl.cpu().numpy()
    del A
    bl = np.array(by_len)
    grid = inf_np[err].reshape(nl, nt)       # layer-major, settled Day 1
    by_pos, by_layer = grid.sum(axis=0), grid.sum(axis=1)
    # D3.10 branch 2 (22-09-2026): BOS error is structurally zero; FINAL-position error is
    # not, and is typically the largest. Position summaries use positions 1..n-1.
    interior = by_pos[1:] if nt > 1 else by_pos
    ni = len(interior)
    etot_pos = by_pos.sum()

    return {
        "replacement_score": rep,
        "completeness_score": comp,
        "error_mass_total": err_inf,
        "mean_path_length": float((np.arange(1, len(bl) + 1) @ bl) / bl.sum()),
        "err_pos_thirds": thirds(interior),
        "err_pos_argmax_norm": float(np.argmax(interior) / max(ni - 1, 1)),
        "err_pos_gini": gini(interior),
        "err_pos_gini_norm": gini(interior) * ni / (ni - 1) if ni > 1 else 0.0,
        "err_second_last_pos": float(by_pos[-2]) if nt > 1 else 0.0,
        "err_layer_thirds": thirds(by_layer),
        "err_layer_argmax": int(np.argmax(by_layer)),
        "err_late_layers": float(by_layer[-6:].sum()),
        "err_bos_pos": float(by_pos[0]),
        "err_final_pos": float(by_pos[-1]),
        "err_final_share": float(by_pos[-1] / etot_pos) if etot_pos > 0 else 0.0,
        "error_by_position": by_pos.tolist(),
        "error_by_layer": by_layer.tolist(),
        "n_nodes": int(N),
        "n_features": n_feat,
        "n_interior": ni,
        "n_edges": int(in_deg.sum()),
        "n_tokens": nt,
        "series_steps": int(steps),
        "node_count_curve": node_count_curve(inf_np),
    }


def validate_against_library(gr, cfg, lib_fn, tol=1e-4):
    """The fast path must reproduce circuit_tracer's compute_graph_scores."""
    lib_rep, lib_comp = lib_fn(gr)
    m = graph_metrics_fast(gr, cfg)
    d_rep = abs(m["replacement_score"] - lib_rep)
    d_comp = abs(m["completeness_score"] - lib_comp)
    print(f"replacement   ours {m['replacement_score']:.6f}  library {lib_rep:.6f}  delta {d_rep:.2e}")
    print(f"completeness  ours {m['completeness_score']:.6f}  library {lib_comp:.6f}  delta {d_comp:.2e}")
    assert d_rep < tol and d_comp < tol, "fast path does not match the library"
    print(f"matches the library within {max(d_rep, d_comp):.2e}")
    return m


def validate_fast(gr, cfg, tol=1e-3):
    """Run both implementations on one graph; they must agree before the fast
    path is trusted for the corpus."""
    slow = graph_metrics(gr, cfg, prune_fn=None)
    fast = graph_metrics_fast(gr, cfg)
    keys = ["replacement_score", "completeness_score", "error_mass_total",
            "mean_path_length", "err_pos_gini", "err_late_layers"]
    worst = 0.0
    for k in keys:
        d = abs(slow[k] - fast[k])
        worst = max(worst, d)
        print(f"{k:22s} slow {slow[k]:.6f}  fast {fast[k]:.6f}  delta {d:.2e}")
    for k in ("n_nodes", "n_features", "n_tokens"):
        assert slow[k] == fast[k], f"{k} differs"
    assert worst < tol, f"implementations disagree by {worst:.2e}"
    print(f"\nagree within {worst:.2e}  (tolerance {tol:.0e})")
    return slow, fast
