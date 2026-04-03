#!/usr/bin/env python3
"""
analyse_e3d_dataset.py
======================
Comprehensive analysis of the e3d_metric HDF5 dataset.

Usage:
    python scripts/analyse_e3d_dataset.py \
        --h5 /media/opervu-user/Data2/ws/data_langgeonet_e3d/e3d_metric_train_ep500.h5 \
        [--max-episodes N]   # analyse only the first N episodes (default: all)
        [--plot]             # save histogram plots alongside the script
"""
from __future__ import annotations

import argparse
import io
import os
import pickle
import time
from collections import Counter, defaultdict

import h5py
import numpy as np

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

    # ── 10. Root causes summary ──────────────────────────────────────────────
    print(sep2)
    print("  10. ROOT CAUSE SUMMARY — WHY NORMALISED COSTS COLLAPSE")
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


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5",  default="/media/opervu-user/Data2/ws/data_langgeonet_e3d/e3d_metric_train_ep500.h5",
                        help="Path to the HDF5 dataset file")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Analyse only the first N episodes (default: all)")
    parser.add_argument("--plot", action="store_true",
                        help="Save matplotlib summary plots")
    args = parser.parse_args()

    analyse(h5_path=args.h5, max_episodes=args.max_episodes, save_plots=args.plot)
