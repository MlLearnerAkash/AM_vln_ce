#!/usr/bin/env python3
"""
analyse_e3d_dataset.py
======================
Comprehensive analysis of the e3d HDF5 dataset.
Handles both the original e3d_metric format and the action-instruction format
(frames containing ``next_action_instruction`` fields).

Usage:
    python utils/analyse_e3d_dataset.py \
        --h5 /media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_test.h5 \
        [--max-episodes N]   # analyse only the first N episodes (default: all)
        [--plot]             # save plots alongside the script
"""
from __future__ import annotations
import argparse
import io
import os
import pickle
import time
from collections import Counter, defaultdict

import re

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# MP3D semantic category map
# ─────────────────────────────────────────────────────────────────────────────
MP3D_CATS = {
    0: "void",        1: "wall",          2: "floor",         3: "chair",
    4: "door",        5: "table",         6: "picture",       7: "cabinet",
    8: "cushion",     9: "window",       10: "sofa",         11: "bed",
   12: "curtain",    13: "chest_of_drawers", 14: "plant",   15: "sink",
   16: "stairs",     17: "ceiling",     18: "toilet",       19: "stool",
   20: "towel",      21: "mirror",      22: "tv_monitor",   23: "shower",
   24: "column",     25: "bathtub",     26: "counter",      27: "fireplace",
   28: "lighting",   29: "beam",        30: "railing",      31: "shelving",
   32: "blinds",     33: "gym_equipment", 34: "seating",   35: "board_panel",
   36: "furniture",  37: "appliances",  38: "clothes",      39: "objects",
}

# Pre-build regex for object names (longest first to avoid partial matches)
_OBJ_NAMES = sorted(MP3D_CATS.values(), key=len, reverse=True)
_OBJ_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(o) for o in _OBJ_NAMES) + r')\b', re.IGNORECASE
)

# Directional categories
_DIR_LEFT    = re.compile(r'\bleft\b',                         re.IGNORECASE)
_DIR_RIGHT   = re.compile(r'\bright\b',                        re.IGNORECASE)
_DIR_FORWARD = re.compile(r'\b(forward|straight|ahead)\b',     re.IGNORECASE)
_DIR_STOP    = re.compile(r'\b(halt|stop|destination|arrived)\b', re.IGNORECASE)


def _parse_instruction(text: str):
    """Return (direction_label, object_name | None) for one instruction string."""
    if _DIR_LEFT.search(text):
        direction = "left"
    elif _DIR_RIGHT.search(text):
        direction = "right"
    elif _DIR_FORWARD.search(text):
        direction = "forward"
    elif _DIR_STOP.search(text):
        direction = "stop"
    else:
        direction = "other"

    obj_match = _OBJ_PATTERN.search(text)
    obj = obj_match.group(1).lower() if obj_match else None
    return direction, obj


# ── optional matplotlib (non-interactive backend) ────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Safe unpickler – skips Habitat / Magnum C-extension modules that may not
# be importable in all environments.
# ─────────────────────────────────────────────────────────────────────────────
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "_magnum" in module or "habitat_sim" in module:
            class _Dummy:
                def __init__(self, *a, **kw): pass
                def __setstate__(self, d):
                    if isinstance(d, dict):
                        self.__dict__.update(d)
            return _Dummy
        return super().find_class(module, name)


def _load_graph(raw_bytes: bytes):
    return _SafeUnpickler(io.BytesIO(raw_bytes)).load()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame cost analysis (mirrors the dataset's overlay_costs logic)
# ─────────────────────────────────────────────────────────────────────────────
def _intra_frame_costs(G, apl, all_nodes, node_to_idx):
    """
    For every frame in the graph, compute:
      - K  : number of objects visible
      - raw_costs : mean row of intra-frame path-length sub-matrix (one per node)
      - normed    : min-max normalised raw_costs (the values shown in the overlay)
      - span      : c_max - c_min  (0.0 means all values collapse to the same colour)

    Returns a list of dicts, one per frame.
    """
    frame_indices = sorted({G.nodes[n]["map"][0] for n in all_nodes})
    results = []
    for fi in frame_indices:
        fnodes = [n for n in all_nodes if G.nodes[n]["map"][0] == fi]
        K = len(fnodes)
        idxs = [node_to_idx[n] for n in fnodes]
        if K == 0:
            continue
        path_rows = apl[np.ix_(idxs, idxs)]          # [K, K]
        raw = np.array([float(np.nanmean(path_rows[k])) for k in range(K)])
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            continue
        c_min, c_max = float(finite.min()), float(finite.max())
        span = c_max - c_min
        if span > 1e-6:
            normed = (raw - c_min) / span
        else:
            normed = np.zeros(K, dtype=np.float32)
        results.append(dict(frame_idx=fi, K=K, raw_costs=raw,
                            normed=normed, span=span,
                            c_min=c_min, c_max=c_max))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyse(h5_path: str, max_episodes: int | None = None, save_plots: bool = False):
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  E3D Dataset Analysis")
    print(f"  File : {h5_path}")
    print(f"{'='*70}\n")

    # ── file-level metadata ──────────────────────────────────────────────────
    file_size_gb = os.path.getsize(h5_path) / 1024**3
    print(f"File size : {file_size_gb:.2f} GB")

    with h5py.File(h5_path, "r") as hf:
        all_ep_keys = sorted(hf.keys())

    n_episodes_total = len(all_ep_keys)
    print(f"Episodes in file : {n_episodes_total}")

    ep_keys = all_ep_keys if max_episodes is None else all_ep_keys[:max_episodes]
    if max_episodes is not None:
        print(f"Analysing first   : {len(ep_keys)} episodes")

    # ─── per-episode accumulators ────────────────────────────────────────────
    frames_per_ep        = []          # int
    rgb_shapes           = Counter()   # (H, W)
    graph_byte_sizes     = []          # bytes (compressed on disk)
    graph_node_counts    = []
    graph_edge_counts    = []
    intra_edge_e3d       = []          # raw intra-frame e3d edge weights (m)
    da_edge_counts       = []
    intra_edge_counts_ep = []

    # per-frame / per-mask
    masks_per_frame      = []          # K values (all frames, all eps)
    frame_cost_spans     = []          # span after intra-frame min-max norm
    all_normed_costs     = []          # every normalised cost value (flat)
    all_raw_costs        = []          # every raw mean path-length value
    k_degenerate         = 0          # frames where K <= 1 or span ≈ 0
    k_nondegenerate      = 0
    frames_k2_zero_span  = 0
    total_frames_ge2     = 0

    # instruction text stats
    instruction_lengths  = []          # word count

    # next_action_instruction stats
    direction_counts     = Counter()   # left / right / forward / stop / other
    object_counts        = Counter()   # MP3D category names
    has_action_instrs    = False

    print("\nProcessing episodes …")
    for ep_i, ep_key in enumerate(ep_keys):
        if ep_i % 50 == 0 and ep_i > 0:
            elapsed = time.time() - t0
            print(f"  {ep_i}/{len(ep_keys)} done  ({elapsed:.0f}s elapsed)")

        with h5py.File(h5_path, "r") as hf:
            ep_grp = hf[ep_key]

            # ── frames ──────────────────────────────────────────────────────
            frame_grps = ep_grp["frames"]
            n_frames = len(frame_grps)
            frames_per_ep.append(n_frames)

            for fk in sorted(frame_grps.keys()):
                fg = frame_grps[fk]
                rgb = fg["rgb"][()]
                rgb_shapes[(rgb.shape[0], rgb.shape[1])] += 1

            # ── instruction ─────────────────────────────────────────────────
            instr_raw = ep_grp["instruction"][()]
            instr = instr_raw.decode("utf-8") if isinstance(instr_raw, bytes) else str(instr_raw)
            instruction_lengths.append(len(instr.split()))

            # ── next_action_instruction (per frame) ─────────────────────────
            for fk in sorted(frame_grps.keys()):
                fg = frame_grps[fk]
                if "next_action_instruction" in fg:
                    has_action_instrs = True
                    nai_raw = fg["next_action_instruction"][()]
                    nai = nai_raw.decode("utf-8") if isinstance(nai_raw, bytes) else str(nai_raw)
                    direction, obj = _parse_instruction(nai)
                    direction_counts[direction] += 1
                    if obj is not None:
                        object_counts[obj] += 1

            # ── graph ───────────────────────────────────────────────────────
            graph_bytes_raw = ep_grp["graph"][()].tobytes()
            graph_byte_sizes.append(len(graph_bytes_raw))

        try:
            G = _load_graph(graph_bytes_raw)
        except Exception as e:
            print(f"  [WARN] episode {ep_key}: failed to load graph ({e})")
            continue

        N = G.number_of_nodes()
        E = G.number_of_edges()
        graph_node_counts.append(N)
        graph_edge_counts.append(E)

        e3d_weights = [(u, v, d.get("e3d", None)) for u, v, d in G.edges(data=True)]
        da_e  = [(u, v, w) for u, v, w in e3d_weights if w is not None and w == 0.0]
        intra = [(u, v, w) for u, v, w in e3d_weights if w is not None and w >  0.0]
        da_edge_counts.append(len(da_e))
        intra_edge_counts_ep.append(len(intra))
        intra_edge_e3d.extend(w for _, _, w in intra)

        # ── per-frame cost analysis ──────────────────────────────────────────
        if N == 0 or "all_paths_lengths" not in G.graph:
            continue

        apl = G.graph["all_paths_lengths"]              # (N, N) float
        all_nodes  = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        frame_results = _intra_frame_costs(G, apl, all_nodes, node_to_idx)

        for fr in frame_results:
            K = fr["K"]
            masks_per_frame.append(K)
            span = fr["span"]
            frame_cost_spans.append(span)

            all_raw_costs.extend(fr["raw_costs"].tolist())
            all_normed_costs.extend(fr["normed"].tolist())

            if K >= 2:
                total_frames_ge2 += 1
                if span < 1e-4:
                    k_degenerate += 1
                    if K == 2:
                        frames_k2_zero_span += 1
                else:
                    k_nondegenerate += 1

    elapsed = time.time() - t0
    print(f"  Done. ({elapsed:.1f}s)\n")

    # ─────────────────────────────────────────────────────────────────────────
    # REPORT
    # ─────────────────────────────────────────────────────────────────────────
    sep  = "─" * 70
    sep2 = "═" * 70

    def P(label, value, unit=""):
        print(f"  {label:<45} {value} {unit}".rstrip())

    # ── 1. Episode & Frame counts ────────────────────────────────────────────
    print(sep2)
    print("  1. EPISODE & FRAME OVERVIEW")
    print(sep2)
    fps = np.array(frames_per_ep)
    P("Episodes analysed",          len(ep_keys))
    P("Total frames",               int(fps.sum()))
    P("Avg frames / episode",       f"{fps.mean():.1f}")
    P("Std frames / episode",       f"{fps.std():.1f}")
    P("Min frames / episode",       int(fps.min()))
    P("Max frames / episode",       int(fps.max()))
    P("Median frames / episode",    int(np.median(fps)))
    print()

    # ── 2. RGB frame dimensions ───────────────────────────────────────────────
    print(sep)
    print("  2. RGB FRAME DIMENSIONS")
    print(sep)
    for (H, W), cnt in sorted(rgb_shapes.items(), key=lambda x: -x[1]):
        P(f"  {H} × {W}  px",  f"{cnt} frames  ({100*cnt/fps.sum():.1f}%)")
    print()

    # ── 3. Graph statistics ──────────────────────────────────────────────────
    print(sep)
    print("  3. GRAPH STATISTICS  (per episode)")
    print(sep)
    gnc = np.array(graph_node_counts)
    gec = np.array(graph_edge_counts)
    dac = np.array(da_edge_counts)
    iac = np.array(intra_edge_counts_ep)
    gbs = np.array(graph_byte_sizes) / 1024**2

    P("Avg nodes / episode",        f"{gnc.mean():.1f}  (std {gnc.std():.1f})")
    P("Avg edges / episode",        f"{gec.mean():.1f}  (std {gec.std():.1f})")
    P("Avg DA edges / episode",     f"{dac.mean():.1f}  ({100*dac.sum()/max(1,gec.sum()):.1f}% of all edges)")
    P("Avg intra-frame edges / ep", f"{iac.mean():.1f}  ({100*iac.sum()/max(1,gec.sum()):.1f}% of all edges)")
    P("Avg graph size (pickled)",   f"{gbs.mean():.2f} MB  (max {gbs.max():.2f})")
    print()

    # ── 4. Intra-frame edge E3D weights ─────────────────────────────────────
    print(sep)
    print("  4. INTRA-FRAME EDGE E3D WEIGHTS  (raw Euclidean 3-D distance, metres)")
    print(sep)
    e3d_arr = np.array(intra_edge_e3d)
    if e3d_arr.size:
        for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
            P(f"  p{p:3d}", f"{np.percentile(e3d_arr, p):.4f} m")
        P("Mean",  f"{e3d_arr.mean():.4f} m")
        P("Std",   f"{e3d_arr.std():.4f} m")
    print()

    # ── 5. Masks per frame ──────────────────────────────────────────────────
    print(sep)
    print("  5. VISIBLE OBJECTS (MASKS) PER FRAME")
    print(sep)
    mpf = np.array(masks_per_frame)
    P("Total frame-object entries",  int(mpf.sum()))
    P("Avg masks / frame",           f"{mpf.mean():.2f}")
    P("Std masks / frame",           f"{mpf.std():.2f}")
    P("Min masks / frame",           int(mpf.min()))
    P("Max masks / frame",           int(mpf.max()))
    P("Median masks / frame",        int(np.median(mpf)))
    print("  Distribution:")
    kc = Counter(mpf.tolist())
    for k in sorted(kc):
        bar = "█" * int(40 * kc[k] / mpf.size)
        print(f"    K={k:3d}  {kc[k]:6d} frames  ({100*kc[k]/mpf.size:5.1f}%)  {bar}")
    print()

    # ── 6. Per-frame cost BEFORE normalization ───────────────────────────────
    print(sep)
    print("  6. PER-FRAME RAW COSTS  (mean intra-frame path length, metres)")
    print(sep)
    rc = np.array(all_raw_costs)
    if rc.size:
        valid_rc = rc[np.isfinite(rc)]
        for p in [0, 5, 25, 50, 75, 95, 100]:
            P(f"  p{p:3d}", f"{np.percentile(valid_rc, p):.4f} m")
        P("Mean", f"{valid_rc.mean():.4f} m")
        P("Std",  f"{valid_rc.std():.4f} m")
    print()

    # ── 7. Intra-frame normalised costs (the values shown in overlay) ────────
    print(sep)
    print("  7. INTRA-FRAME MIN-MAX NORMALISED COSTS  (what the visualizer shows)")
    print(sep)
    nc = np.array(all_normed_costs)
    if nc.size:
        for p in [0, 5, 25, 50, 75, 95, 100]:
            P(f"  p{p:3d}", f"{np.percentile(nc, p):.4f}")
        P("Mean", f"{nc.mean():.4f}")
        P("Std",  f"{nc.std():.4f}")
        # histogram buckets
        hist, edges = np.histogram(nc, bins=10, range=(0, 1))
        print("  Histogram (0→1):")
        for i in range(len(hist)):
            bar = "█" * int(40 * hist[i] / hist.max())
            print(f"    [{edges[i]:.1f}–{edges[i+1]:.1f}]  {hist[i]:8d}  {bar}")
    print()

    # ── 8. Cost SPAN per frame (discriminability) ────────────────────────────
    print(sep)
    print("  8. COST SPAN PER FRAME  (c_max − c_min, higher = more discriminable)")
    print(sep)
    cs = np.array(frame_cost_spans)
    for p in [0, 5, 25, 50, 75, 95, 100]:
        P(f"  p{p:3d}", f"{np.percentile(cs, p):.4f}")
    P("Mean span", f"{cs.mean():.4f}")
    P("Std span",  f"{cs.std():.4f}")
    print()
    print("  Degeneracy (span < 1e-4, i.e. all masks same colour):")
    P("  Frames with K ≥ 2",               total_frames_ge2)
    P("  Degenerate frames (span ≈ 0)",    f"{k_degenerate}  ({100*k_degenerate/max(1,total_frames_ge2):.1f}%)")
    P("  Non-degenerate frames",           f"{k_nondegenerate}  ({100*k_nondegenerate/max(1,total_frames_ge2):.1f}%)")
    P("  Degenerate due to K=2",           f"{frames_k2_zero_span}  ({100*frames_k2_zero_span/max(1,k_degenerate):.1f}% of degenerate)")
    print()

    # ── 9. Instruction text statistics ──────────────────────────────────────
    print(sep)
    print("  9. INSTRUCTION TEXT")
    print(sep)
    il = np.array(instruction_lengths)
    P("Avg words / instruction",  f"{il.mean():.1f}")
    P("Std",                      f"{il.std():.1f}")
    P("Min words",                int(il.min()))
    P("Max words",                int(il.max()))
    print()

    # ── 10. Next-action instruction analysis ─────────────────────────────────
    if has_action_instrs:
        print(sep)
        print("  10. NEXT-ACTION INSTRUCTION ANALYSIS")
        print(sep)
        total_nai = sum(direction_counts.values())
        P("Total next_action_instruction entries", total_nai)
        print()
        print("  Directional distribution:")
        for d in ["forward", "left", "right", "stop", "other"]:
            cnt = direction_counts.get(d, 0)
            bar = "█" * int(40 * cnt / max(1, total_nai))
            P(f"    {d:<10}", f"{cnt:6d}  ({100*cnt/max(1,total_nai):5.1f}%)  {bar}")
        print()
        print("  Object reference distribution (top 20 by frequency):")
        top_objs = object_counts.most_common(20)
        max_obj_cnt = top_objs[0][1] if top_objs else 1
        for obj, cnt in top_objs:
            bar = "█" * int(40 * cnt / max(1, max_obj_cnt))
            P(f"    {obj:<22}", f"{cnt:6d}  ({100*cnt/max(1,total_nai):5.1f}%)  {bar}")
        print()

    # ── 11. Root causes summary ──────────────────────────────────────────────
    print(sep2)
    print("  11. ROOT CAUSE SUMMARY — WHY NORMALISED COSTS COLLAPSE")
    print(sep2)
    all_edges = sum(graph_edge_counts)
    all_da    = sum(da_edge_counts)
    all_intra = sum(intra_edge_counts_ep)
    k2_frac   = 100 * kc.get(2, 0) / max(1, mpf.size)
    print(f"""
  Cause 1 – K=2 symmetry ({k2_frac:.1f}% of frames have only 2 objects)
    When K=2 the intra-frame path-length sub-matrix is [[0, d],[d, 0]].
    Both row means equal d/2, so c_min == c_max and span = 0.
    Min-max normalisation then collapses every object to 0.0.

  Cause 2 – DA loop-closure edges with e3d=0 ({100*all_da/max(1,all_edges):.1f}% of all edges)
    {all_da} DA edges vs {all_intra} intra-frame edges.
    Dijkstra routes nearly every inter-node path through zero-cost DA
    shortcuts, compressing the all-pairs path-length range and reducing
    the discriminability of the intra-frame sub-matrix.

  Cause 3 – Intra-frame sub-matrix diagonal is always 0
    Even for K>2 the diagonal zeros anchor every row mean, making all
    raw costs converge toward 0.5 × (sum of one row) and reducing span.

  Degenerate rate : {100*k_degenerate/max(1,total_frames_ge2):.1f}% of frames with K≥2 have span ≈ 0.
""")

    print(f"  Total wall-clock time: {time.time()-t0:.1f}s")
    print(sep2 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIONAL PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    if save_plots and HAS_MPL:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        _plot(mpf, nc, cs, e3d_arr, fps, out_dir)
        if has_action_instrs:
            _plot_action_instrs(direction_counts, object_counts, out_dir)
    elif save_plots and not HAS_MPL:
        print("  [INFO] matplotlib not available – skipping plots.")


def _plot(mpf, nc, cs, e3d_arr, fps, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("E3D Dataset Analysis", fontsize=14, fontweight="bold")

    def _hist(ax, data, bins, title, xlabel, color="steelblue", logy=False, hist_range=None):
        kwargs = dict(color=color, edgecolor="white", linewidth=0.3)
        if hist_range is not None:
            kwargs["range"] = hist_range
        ax.hist(data, bins=bins, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)

    _hist(axes[0, 0], fps,    bins=30, title="Frames per Episode",
          xlabel="# frames", color="steelblue")

    kvals = list(mpf)
    _hist(axes[0, 1], kvals, bins=range(0, int(max(kvals)) + 2),
          title="Visible Masks per Frame (K)",
          xlabel="K", color="darkorange")

    _hist(axes[0, 2], e3d_arr, bins=50,
          title="Intra-frame E3D Edge Weights (raw, m)",
          xlabel="distance (m)", color="seagreen")

    _hist(axes[1, 0], nc,  bins=50, hist_range=(0, 1),
          title="Normalised Costs (what overlay shows)",
          xlabel="normalised cost", color="crimson")
    axes[1, 0].axvline(nc.mean(), color="black", linestyle="--",
                       linewidth=1, label=f"mean={nc.mean():.2f}")
    axes[1, 0].legend(fontsize=8)

    _hist(axes[1, 1], cs, bins=50,
          title="Cost Span per Frame (c_max − c_min)",
          xlabel="span (m)", color="mediumpurple", logy=True)
    axes[1, 1].axvline(1e-4, color="red", linestyle="--",
                       linewidth=1, label="degenerate threshold")
    axes[1, 1].legend(fontsize=8)

    # Mask distribution pie
    from collections import Counter
    kc = Counter(mpf.tolist())
    top_k = sorted(kc.items())[:8]
    labels = [f"K={k}" for k, _ in top_k]
    sizes  = [v for _, v in top_k]
    axes[1, 2].pie(sizes, labels=labels, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 8})
    axes[1, 2].set_title("Mask Count Distribution", fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "e3d_dataset_analysis.png")
    fig.savefig(out_path, dpi=150)
    print(f"  Plots saved → {out_path}")
    plt.close(fig)


def _plot_action_instrs(direction_counts: Counter, object_counts: Counter, out_dir: str):
    """Pie chart for directional cues + bar plot for object references."""
    dir_order  = ["forward", "left", "right", "stop", "other"]
    dir_colors = ["#4C9BE8", "#F4A261", "#E76F51", "#2A9D8F", "#B5B5B5"]

    dir_labels = []
    dir_sizes  = []
    dir_cols   = []
    for d, c in zip(dir_order, dir_colors):
        cnt = direction_counts.get(d, 0)
        if cnt > 0:
            dir_labels.append(d)
            dir_sizes.append(cnt)
            dir_cols.append(c)

    # Object bar (sort by frequency, all objects)
    obj_items = object_counts.most_common()
    obj_names = [o for o, _ in obj_items]
    obj_vals  = [v for _, v in obj_items]

    fig, (ax_pie, ax_bar) = plt.subplots(
        1, 2, figsize=(16, max(6, len(obj_names) * 0.35 + 2))
    )
    fig.suptitle("Next-Action Instruction Analysis", fontsize=14, fontweight="bold")

    # — Pie chart ────────────────────────────────────────────────────────────
    wedges, texts, autotexts = ax_pie.pie(
        dir_sizes,
        labels=dir_labels,
        colors=dir_cols,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax_pie.set_title("Directional Cue Distribution", fontsize=12)
    total = sum(dir_sizes)
    ax_pie.legend(
        [f"{l}  ({c:,} / {100*c/total:.1f}%)" for l, c in zip(dir_labels, dir_sizes)],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=9,
        frameon=False,
    )

    # — Bar chart ─────────────────────────────────────────────────────────────
    y_pos = np.arange(len(obj_names))
    bars = ax_bar.barh(y_pos, obj_vals, color="steelblue", edgecolor="white", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(obj_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Count", fontsize=10)
    ax_bar.set_title("Object Reference Frequency (MP3D categories)", fontsize=12)
    ax_bar.grid(axis="x", alpha=0.3)
    # annotate counts
    for bar, val in zip(bars, obj_vals):
        ax_bar.text(
            bar.get_width() + max(obj_vals) * 0.005, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=8,
        )

    fig.tight_layout()
    out_path = os.path.join(out_dir, "action_instruction_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Action instruction plots saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5",  default="/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_test.h5",
                        help="Path to the HDF5 dataset file")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Analyse only the first N episodes (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Save matplotlib summary plots")
    args = parser.parse_args()

    analyse(h5_path=args.h5, max_episodes=args.max_episodes, save_plots=args.plot)
