"""circuit_view - two readable views of a circuit-tracer attribution graph.

    circuit_map(g, model, n_layers)   position x layer picture of the whole graph
    trace(g, model, n_layers)         backward walk from the predicted logit

Both take a Graph straight from `attribute(...)` or `Graph.from_pt(...)`.

Node layout assumed (as circuit-tracer builds it):
    [features | errors (n_layers * n_tok, layer-major) | tokens | logits]
Feature metadata is `active_features[selected_features[i]] -> (layer, position, feature_id)`.
"""
import numpy as np

try:
    import torch
except ImportError:                                   # plotting-only environments
    torch = None


# --------------------------------------------------------------------------- helpers

def _np(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _influence(adj, logit_probs, n_logits):
    """Probability-weighted path strength from every node to the logits.

    Uses the GPU when the adjacency is a CUDA tensor, otherwise NumPy. The graph is a
    DAG, so the backward series terminates exactly.
    """
    if torch is not None and torch.is_tensor(adj) and adj.is_cuda:
        A = adj.abs().float()
        A = A / A.sum(1, keepdim=True).clamp_min(1e-30)
        w = torch.zeros(A.shape[0], device=A.device)
        w[-n_logits:] = torch.as_tensor(_np(logit_probs), device=A.device,
                                        dtype=torch.float32)
        v, infl = w.clone(), torch.zeros_like(w)
        for _ in range(500):
            v = v @ A
            infl += v
            if float(v.sum()) < 1e-12:
                break
        return A.cpu().numpy(), infl.cpu().numpy()

    A = np.abs(_np(adj)).astype(np.float32)
    rs = A.sum(1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)
    w = np.zeros(A.shape[0], dtype=np.float32)
    w[-n_logits:] = _np(logit_probs)
    v, infl = w.copy(), np.zeros_like(w)
    for _ in range(500):
        v = v @ A
        infl += v
        if v.sum() < 1e-12:
            break
    return A, infl


def _unpack(g, model, n_layers):
    af = _np(g.active_features).astype(int)           # (n_active, 3) layer, pos, feature id
    sel = _np(g.selected_features).astype(int)        # indexes into active_features
    meta = af[sel]                                    # (n_feat, 3)
    n_feat = len(sel)
    toks = [model.tokenizer.decode([int(t)]) for t in _np(g.input_tokens).astype(int)]
    nt = len(toks)
    probs = _np(g.logit_probabilities)
    A, infl = _influence(g.adjacency_matrix, probs, len(probs))
    return dict(A=A, infl=infl, meta=meta, n_feat=n_feat, nt=nt, nl=n_layers, toks=toks,
                probs=probs, err0=n_feat, tok0=n_feat + n_layers * nt)


def _predicted(g, model):
    probs = _np(g.logit_probabilities)
    k = int(np.argmax(probs))
    for name in ("logit_token_ids", "logit_tokens", "logit_targets"):
        if hasattr(g, name):
            try:
                tid = int(np.atleast_1d(_np(getattr(g, name))[k]).ravel()[-1])
                return model.tokenizer.decode([tid]), float(probs[k])
            except Exception:
                continue
    return "?", float(probs[k])


def _label(i, d):
    if i < d["n_feat"]:
        lay, pos, fid = d["meta"][i]
        return f"L{lay:02d} pos{pos}({d['toks'][pos]!r}) #{fid}"
    if i < d["tok0"]:
        j = i - d["err0"]
        p = j % d["nt"]
        return f"ERROR L{j // d['nt']:02d} pos{p}({d['toks'][p]!r})"
    if i < d["tok0"] + d["nt"]:
        p = i - d["tok0"]
        return f"TOKEN pos{p} {d['toks'][p]!r}"
    return "LOGIT"


# --------------------------------------------------------------------------- the map

def circuit_map(g, model, n_layers, top_features=40, n_arrows=14, label_top=6,
                figsize=None, ax=None, title=None):
    """Position x layer picture of one attribution graph.

    blue circles  features, placed at (token position, layer), area = influence
    red shading   error-node influence: where the transcoder could not reconstruct
    black arrows  the strongest CROSS-POSITION feature edges - information moving,
                  which is where the mechanism lives
    grey column   the read-out position, which the logits read from
    """
    import matplotlib.pyplot as plt

    d = _unpack(g, model, n_layers)
    infl, meta, n_feat, nt, nl = d["infl"], d["meta"], d["n_feat"], d["nt"], d["nl"]

    err_grid = infl[d["err0"]:d["tok0"]].reshape(nl, nt)
    f_inf = infl[:n_feat]
    f_lay, f_pos = meta[:, 0], meta[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(7, 1.05 * nt + 3), 7))

    ax.imshow(err_grid, cmap="Reds", origin="lower", aspect="auto",
              extent=(-0.5, nt - 0.5, -0.5, nl - 0.5),
              vmin=0, vmax=max(err_grid.max(), 1e-9))
    ax.axvspan(nt - 1.5, nt - 0.5, color="0.5", alpha=0.12, zorder=0)

    keep = np.argsort(-f_inf)[:top_features]
    keep = keep[f_inf[keep] > 0]
    if len(keep):
        ax.scatter(f_pos[keep], f_lay[keep],
                   s=40 + 4000 * f_inf[keep] / max(f_inf.max(), 1e-12),
                   c="#1d4ed8", alpha=0.65, edgecolors="white", linewidths=0.6, zorder=3)

    # strongest cross-position edges among the most influential features
    A = d["A"]
    edges = []
    for i in np.argsort(-f_inf)[:top_features]:
        row = A[i]
        for s in np.argsort(-row)[:4]:
            if s >= n_feat or row[s] <= 0:
                continue
            if meta[s, 1] != meta[i, 1]:              # crosses a token position
                edges.append((float(row[s] * f_inf[i]), int(s), int(i)))
    edges.sort(reverse=True)
    seen = set()
    for _, s, i in edges[:n_arrows]:
        key = (meta[s, 1], meta[s, 0], meta[i, 1], meta[i, 0])
        if key in seen:
            continue
        seen.add(key)
        ax.annotate("", xy=(meta[i, 1], meta[i, 0]), xytext=(meta[s, 1], meta[s, 0]),
                    arrowprops=dict(arrowstyle="-|>", color="#111827", lw=1.3,
                                    alpha=0.75, shrinkA=6, shrinkB=6,
                                    connectionstyle="arc3,rad=0.12"), zorder=4)

    for i in np.argsort(-f_inf)[:label_top]:
        ax.annotate(f"#{meta[i, 2]}", (f_pos[i], f_lay[i]), fontsize=7, color="#1e3a8a",
                    xytext=(5, 5), textcoords="offset points", zorder=5)

    ax.set_xticks(range(nt))
    ax.set_xticklabels([t.replace("\n", "\\n") for t in d["toks"]], rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticks(range(0, nl, 2))
    ax.set_ylabel("layer")
    ax.set_xlim(-0.5, nt - 0.5)
    ax.set_ylim(-0.5, nl - 0.5)

    pred, p = _predicted(g, model)
    if title is None:
        title = f"{getattr(g, 'input_string', '')!r}\npredicts {pred!r}  (p={p:.2f})"
    ax.set_title(title, fontsize=10, loc="left")

    # one-line orientation summary under the plot
    tok_inf = infl[d["tok0"]:d["tok0"] + nt]
    third = nl // 3
    early = f_inf[f_lay < third].sum()
    mid = f_inf[(f_lay >= third) & (f_lay < 2 * third)].sum()
    late = f_inf[f_lay >= 2 * third].sum()
    ax.set_xlabel(
        f"token influence peak: {d['toks'][int(np.argmax(tok_inf))]!r}   |   "
        f"error peak: {d['toks'][int(np.argmax(err_grid.sum(0)))]!r} "
        f"({err_grid.sum(0).max() / max(err_grid.sum(), 1e-12):.0%} of error)   |   "
        f"feature influence  early {early:.2f}  mid {mid:.2f}  late {late:.2f}",
        fontsize=8)
    return ax


# ------------------------------------------------------------------------- the trace

def summary(g, model, n_layers, _d=None):
    """Orientation block: prediction, token influence, error by position, layer thirds.

    Read this BEFORE the diagram. It answers, in four lines: did the model do the task,
    where did the answer come from, where did the method go blind, and did anything
    abstract happen in the middle layers.
    """
    d = _d or _unpack(g, model, n_layers)
    infl, meta, n_feat, nt, nl, toks = (d["infl"], d["meta"], d["n_feat"],
                                        d["nt"], d["nl"], d["toks"])
    pred, p = _predicted(g, model)
    tok_inf = infl[d["tok0"]:d["tok0"] + nt]
    err_pos = infl[d["err0"]:d["tok0"]].reshape(nl, nt).sum(0)
    f_lay, f_inf = meta[:, 0], infl[:n_feat]
    t3 = nl // 3
    early, mid, late = (f_inf[f_lay < t3].sum(),
                        f_inf[(f_lay >= t3) & (f_lay < 2 * t3)].sum(),
                        f_inf[f_lay >= 2 * t3].sum())
    peak_tok = toks[int(np.argmax(tok_inf))]
    peak_err = toks[int(np.argmax(err_pos))]
    return "\n".join([
        f"prompt   : {getattr(g, 'input_string', '')!r}",
        f"predicts : {pred!r}  (p={p:.2f})",
        "token influence  : " + ", ".join(f"{toks[i]!r} {tok_inf[i]:.3f}" for i in range(nt)),
        "error by position: " + ", ".join(f"{toks[i]!r} {err_pos[i]:.3f}" for i in range(nt)),
        f"feature influence: early {early:.3f}  mid {mid:.3f}  late {late:.3f}"
        f"   (mid/early {mid / max(early, 1e-12):.2f})",
        f"peak token influence {peak_tok!r} | peak error {peak_err!r} | "
        f"read-out position holds {err_pos[-1] / max(err_pos.sum(), 1e-12):.0%} of all error",
    ])


def describe(g, model, n_layers, depth=4, width=4, min_w=0.01, keep_min=3, labels=None):
    """The whole graph as text: prediction, orientation summaries, and a deduped
    backward trace from the logit. Returned as a string so it can be printed and
    also stored (e.g. for a second rater)."""
    d = _unpack(g, model, n_layers)
    A, probs, meta = d["A"], d["probs"], d["meta"]
    n_feat, nt, nl, toks = d["n_feat"], d["nt"], d["nl"], d["toks"]
    infl = d["infl"]
    labels = labels or {}
    pred, p = _predicted(g, model)
    L = summary(g, model, n_layers, _d=d).split("\n")
    L.append("")

    def lab(i):
        base = _label(i, d)
        if i < n_feat:
            key = (int(meta[i, 0]), int(meta[i, 2]))
            if key in labels and labels[key]:
                return f"{base}  \u00ab{labels[key][:48]}\u00bb"
        return base

    k = int(np.argmax(probs))
    logit = A.shape[0] - len(probs) + k
    expanded, cross = set(), 0
    L.append("")
    L.append(f"LOGIT {pred!r}")

    def walk(i, left, pre):
        nonlocal cross
        if left == 0 or d["err0"] <= i < d["tok0"] + nt:
            return
        for s, w in _top_inputs(A[i], width, min_w, keep_min):
            both = s < n_feat and i < n_feat
            crosses = both and meta[s, 1] != meta[i, 1]
            cross += crosses
            seen = s in expanded and s < n_feat   # only features get expanded
            L.append(f"{pre}<-{' *' if crosses else '  '}{w:.3f}  {lab(s)}"
                     + ("   (expanded above)" if seen else ""))
            if not seen:
                expanded.add(s)
                walk(s, left - 1, pre + "    ")

    walk(logit, depth, "  ")
    L.append("")
    L.append(f"* = crosses a token position ({cross} such edges drawn)")
    return "\n".join(L)


def trace(g, model, n_layers, depth=4, width=4, min_w=0.01, keep_min=3, labels=None):
    print(describe(g, model, n_layers, depth, width, min_w, keep_min, labels))


# ------------------------------------------------------------------- path diagram

def _top_inputs(row, width, min_w, keep_min):
    """Top-`width` inputs of a node. `min_w` only trims BEYOND the first `keep_min`.

    Row weights are normalised per node, so a high fan-in node (the logit has
    thousands of incoming edges) has tiny weights throughout - a fixed threshold
    there silently empties the graph.
    """
    order = np.argsort(-row)[:width]
    out = []
    for rank, s in enumerate(order):
        w = float(row[s])
        if w <= 0:
            break
        if rank < keep_min or w >= min_w:
            out.append((int(s), w))
    return out


def _back_edges(d, depth=4, width=4, min_w=0.01, keep_min=3):
    """Backward-reachable subgraph from the predicted logit: (src, dst, weight)."""
    A, probs = d["A"], d["probs"]
    k = int(np.argmax(probs))
    logit = A.shape[0] - len(probs) + k
    edges, seen = [], set()

    def walk(i, left):
        if left == 0 or d["err0"] <= i < d["tok0"] + d["nt"]:
            return
        for s, w in _top_inputs(A[i], width, min_w, keep_min):
            edges.append((s, i, w))
            if (s, left) not in seen:
                seen.add((s, left))
                walk(s, left - 1)

    walk(logit, depth)
    return logit, edges


def path_diagram(g, model, n_layers, depth=4, width=4, min_w=0.01, keep_min=3,
                 labels=None, figsize=None, ax=None):
    """Only the nodes that feed the answer, placed by token position and layer.

    Every node is labelled. Edge thickness is the normalised attribution weight.
    Bold black edges cross a token position - that is information moving, and it is
    where the mechanism lives. `labels` may be a {(layer, feature_id): text} dict,
    e.g. from Neuronpedia, in which case the text replaces the bare feature id.
    """
    import matplotlib.pyplot as plt

    d = _unpack(g, model, n_layers)
    meta, n_feat, nt, nl, toks = d["meta"], d["n_feat"], d["nt"], d["nl"], d["toks"]
    act = _np(g.activation_values) if hasattr(g, "activation_values") else None
    logit, edges = _back_edges(d, depth, width, min_w, keep_min)

    def coord(i):
        if i < n_feat:
            return float(meta[i, 1]), float(meta[i, 0])
        if i < d["tok0"]:
            j = i - d["err0"]
            return float(j % nt), float(j // nt)
        if i < d["tok0"] + nt:
            return float(i - d["tok0"]), -2.0
        return float(nt - 1), float(nl + 2)

    def text(i):
        if i < n_feat:
            lay, pos, fid = meta[i]
            a = f"\nact {act[i]:.0f}" if act is not None and i < len(act) else ""
            if labels and (int(lay), int(fid)) in labels:
                return f"L{lay} {labels[(int(lay), int(fid))][:34]}{a}"
            return f"L{lay} #{fid}{a}"
        if i < d["tok0"]:
            j = i - d["err0"]
            return f"ERROR L{j // nt}"
        if i < d["tok0"] + nt:
            return f"{toks[i - d['tok0']]!r}"
        return "LOGIT"

    nodes = sorted({s for s, _, _ in edges} | {t for _, t, _ in edges} | {logit})
    # spread nodes that share a cell so labels do not collide
    cells, xy = {}, {}
    for i in nodes:
        x, y = coord(i)
        k = (round(x), round(y))
        cells.setdefault(k, []).append(i)
    for (x, y), group in cells.items():
        for r, i in enumerate(group):
            xy[i] = (x + 0.30 * (r - (len(group) - 1) / 2), y + 0.55 * (r % 2))

    # push apart nodes that share a token column, so the labels stay legible
    MIN_GAP = max(2.0, nl / 12)
    for x0 in {round(v[0]) for v in xy.values()}:
        col = sorted([i for i in xy if round(xy[i][0]) == x0], key=lambda i: xy[i][1])
        for a, b in zip(col, col[1:]):
            if xy[b][1] - xy[a][1] < MIN_GAP:
                xy[b] = (xy[b][0], xy[a][1] + MIN_GAP)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(9, 1.45 * nt), 10))

    for s, t, w in edges:
        if s not in xy or t not in xy:
            continue
        crosses = (s < n_feat and t < n_feat and meta[s, 1] != meta[t, 1])
        ax.annotate("", xy=xy[t], xytext=xy[s], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", lw=0.8 + 4 * w,
                                    color="#111827" if crosses else "#9ca3af",
                                    alpha=0.9 if crosses else 0.55,
                                    shrinkA=12, shrinkB=12,
                                    connectionstyle="arc3,rad=0.1"))

    for i in nodes:
        x, y = xy[i]
        if i >= d["tok0"] + nt:
            fc, ec = "#bbf7d0", "#15803d"
        elif i >= d["tok0"]:
            fc, ec = "#e5e7eb", "#6b7280"
        elif i >= d["err0"]:
            fc, ec = "#fecaca", "#b91c1c"
        else:
            fc, ec = "#dbeafe", "#1d4ed8"
        ax.annotate(text(i), (x, y), ha="center", va="center", fontsize=7.5, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.0))

    ax.set_xticks(range(nt))
    ax.set_xticklabels([t.replace("\n", "\\n") for t in toks], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(list(range(0, nl, 4)) + [-2, nl + 2])
    ax.set_yticklabels([str(v) for v in range(0, nl, 4)] + ["token", "logit"], fontsize=8)
    ax.set_xlim(-0.8, nt - 0.2); ax.set_ylim(-3.5, max(nl + 3.5, max(v[1] for v in xy.values()) + 2))
    ax.set_ylabel("layer"); ax.grid(axis="x", alpha=0.15)
    pred, p = _predicted(g, model)
    ax.set_title(f"{getattr(g, 'input_string', '')!r}  ->  {pred!r} (p={p:.2f})\n"
                 f"blue feature, red error, grey token, green logit; "
                 f"bold edge = crosses a token position", fontsize=9, loc="left")
    return ax


# ------------------------------------------------------- Neuronpedia feature labels

NP_TEMPLATE = ("https://www.neuronpedia.org/api/feature/"
               "gemma-2-2b/{layer}-gemmascope-transcoder-16k/{fid}")
NP_CACHE = "/kaggle/working/np_labels.json"


def _load_cache(path=None):
    import json, os
    path = path or NP_CACHE
    if os.path.exists(path):
        try:
            return {tuple(int(x) for x in k.split("_")): v
                    for k, v in json.load(open(path)).items()}
        except Exception:
            pass
    return {}


def _save_cache(cache, path=None):
    import json, os
    path = path or NP_CACHE
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump({f"{l}_{f}": v for (l, f), v in cache.items()}, open(path, "w"))


def fetch_labels(pairs, template=None, cache_path=None, pause=0.15,
                 timeout=10, verbose=False):
    """Feature descriptions from Neuronpedia for [(layer, feature_id), ...].

    Cached on disk, so each feature is fetched once per session. Failures are cached
    as None and not retried; a missing label never blocks a rating.
    """
    import time as _t
    import requests

    template = template or NP_TEMPLATE
    cache = _load_cache(cache_path)
    todo = [p for p in set(pairs) if p not in cache]
    for lay, fid in todo:
        try:
            r = requests.get(template.format(layer=int(lay), fid=int(fid)), timeout=timeout)
            if r.ok:
                exps = r.json().get("explanations") or []
                cache[(lay, fid)] = (exps[0].get("description") if exps else None)
            else:
                cache[(lay, fid)] = None
        except Exception as ex:
            if verbose:
                print("label fetch failed", lay, fid, type(ex).__name__)
            cache[(lay, fid)] = None
        _t.sleep(pause)
    if todo:
        try:
            _save_cache(cache, cache_path)
        except Exception as ex:
            if verbose:
                print("cache write failed:", type(ex).__name__)
    return {k: v for k, v in cache.items() if v}


def auto_labels(g, model, n_layers, depth=4, width=4, min_w=0.01, keep_min=3, **kw):
    """Fetch labels for exactly the features the path diagram will draw."""
    d = _unpack(g, model, n_layers)
    _, edges = _back_edges(d, depth, width, min_w, keep_min)
    ids = {i for e in edges for i in (e[0], e[1]) if i < d["n_feat"]}
    pairs = [(int(d["meta"][i, 0]), int(d["meta"][i, 2])) for i in ids]
    return fetch_labels(pairs, **kw)


def explain(g, model, n_layers, depth=4, width=4, min_w=0.01, keep_min=3,
            labels=None, show=True, diagram=True):
    """Labelled path diagram plus the text description - the view to rate from.

    Returns {"text": ..., "labels": ...} so the text can be stored for a second rater.
    """
    import matplotlib.pyplot as plt
    if labels is None:
        try:
            labels = auto_labels(g, model, n_layers, depth, width, min_w, keep_min)
        except Exception as ex:
            print(f"(no labels: {type(ex).__name__}; is notebook internet on?)")
            labels = {}
    d = _unpack(g, model, n_layers)
    print(summary(g, model, n_layers, _d=d))          # orientation BEFORE the picture
    print()
    if diagram:
        path_diagram(g, model, n_layers, depth=depth, width=width, min_w=min_w,
                     keep_min=keep_min, labels=labels)
        if show:
            plt.show()
    text = describe(g, model, n_layers, depth, width, min_w, keep_min, labels)
    print(text[text.index("LOGIT "):] if "LOGIT " in text else text)
    named = sum(1 for v in labels.values() if v)
    print(f"\n{named} of the drawn features have a Neuronpedia description.")
    return {"text": text, "labels": {f"{l}_{f}": v for (l, f), v in labels.items()}}
