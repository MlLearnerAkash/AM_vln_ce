import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image
import cv2
from tqdm import tqdm
import pickle
import os
import cv2
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools import mask as mask_utils

def show_frame_pathlengths_heatmap(episode_dir, query_frame_idx):
    """
    For a given frame, show a single image with ALL masks in that frame
    overlaid with a heatmap color = average path length from that mask
    to all other nodes in the graph.

    Parameters
    ----------
    episode_dir      : str   path to episode directory
    query_frame_idx  : int   frame to visualize
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = os.path.join(episode_dir, "nodes_graphObject_4_gt_topometric.pickle")
    if not os.path.exists(graph_path):
        picks = [f for f in os.listdir(episode_dir) if f.endswith(".pickle")]
        if not picks:
            raise FileNotFoundError(f"No pickle found in {episode_dir}")
        graph_path = os.path.join(episode_dir, picks[0])

    G         = pickle.load(open(graph_path, "rb"))
    all_paths = G.graph['all_paths_lengths']   # (N, N)
    all_nodes = list(G.nodes())

    # ── Find nodes in this frame only ────────────────────────────────────────
    frame_nodes = [n for n in all_nodes if G.nodes[n]['map'][0] == query_frame_idx]
    if not frame_nodes:
        valid = sorted(set(G.nodes[n]['map'][0] for n in all_nodes))
        raise ValueError(f"Frame {query_frame_idx} not found in map. Valid frames: {valid}")

    print(f"Frame {query_frame_idx} has {len(frame_nodes)} nodes: {frame_nodes}")

    # ── Load RGB ──────────────────────────────────────────────────────────────
    img_path = os.path.join(episode_dir, f"frame_{query_frame_idx:03d}", "rgb.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"RGB not found: {img_path}")
    rgb = cv2.imread(img_path)[:, :, ::-1]
    H, W = rgb.shape[:2]

    # ── Compute per-node average path length (to all OTHER nodes) ─────────────
    node_distances_raw = {}
    for n in frame_nodes:
        row_idx            = all_nodes.index(n)
        dists              = all_paths[row_idx].copy()   # (N,)
        dists[row_idx]     = 1e6                         # exclude self
        finite             = dists[dists < 1e5]
        node_distances_raw[n] = float(finite.mean()) if len(finite) > 0 else float('inf')

    # normalize to [0, 1] within this frame only
    raw_vals = np.array([node_distances_raw[n] for n in frame_nodes], dtype=np.float32)
    finite_mask = raw_vals < 1e5

    if finite_mask.sum() > 1:   
        f_min = raw_vals[finite_mask].min()
        f_max = raw_vals[finite_mask].max()
        denom = (f_max - f_min) if (f_max - f_min) > 1e-8 else 1.0  # avoid div by zero
        normalized = (raw_vals - f_min) / denom                       # [0, 1]
        normalized[~finite_mask] = float('inf')                       # keep unreachable as inf
    elif finite_mask.sum() == 1:
        normalized = np.where(finite_mask, 0.0, float('inf'))         # single node → 0
    else:
        normalized = np.full(len(frame_nodes), float('inf'))           # all unreachable

    node_distances = {n: float(normalized[i]) for i, n in enumerate(frame_nodes)}

    print("  Per-node mean distances:")
    for n, d in node_distances.items():
        cat = G.nodes[n].get('category_name', '?')
        print(f"    node {n:4d}  ({cat:20s})  mean_dist = {d:.4f}")


    norm  = Normalize(vmin=0.0, vmax=1.0)
    cmap  = cm.get_cmap('RdYlGn_r')   # green=close, red=far

    # ── Build overlay ─────────────────────────────────────────────────────────
    overlay = rgb.copy().astype(np.float32)

    def decode_mask(rle, H, W):
        compressed = mask_utils.frPyObjects(rle, rle['size'][0], rle['size'][1])
        return mask_utils.decode(compressed).astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Frame {query_frame_idx}  —  mask heatmap by mean path length to all other nodes\n"
        f"green = close to rest of graph,  red = far,  grey = unreachable",
        fontsize=11
    )

    label_data = []   # collect for legend

    for n in frame_nodes:
        dist = node_distances[n]
        rle  = G.nodes[n]['segmentation']

        try:
            mask = decode_mask(rle, H, W)
        except Exception:
            continue

        if not mask.any():
            continue

        if dist >= 1e5:
            mask_color = np.array([160, 160, 160], dtype=np.float32)
            alpha      = 0.55
            hex_color  = '#a0a0a0'
        else:
            rgba       = cmap(norm(dist))
            mask_color = np.array(rgba[:3]) * 255
            alpha      = 0.65
            hex_color  = '#%02x%02x%02x' % tuple((np.array(rgba[:3]) * 255).astype(int))

        overlay[mask] = (
            (1 - alpha) * overlay[mask] + alpha * mask_color
        ).clip(0, 255)

        # white contour
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay.astype(np.uint8), cnts, -1, (255, 255, 255), 2)

        # centroid label
        ys, xs = np.where(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        cat    = G.nodes[n].get('category_name', '')[:10]
        label  = f"{dist:.2f}" if dist < 1e5 else "∞"

        axes[1].text(cx, cy, f"n{n}\n{label}\n{cat}",
                     fontsize=6, color='white', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5, lw=0))

        label_data.append((n, cat, dist, hex_color))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Raw Frame {query_frame_idx}", fontsize=10)
    axes[0].axis('off')

    # ── Right: heatmap overlay ────────────────────────────────────────────────
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"Path-length heatmap  ({len(frame_nodes)} masks)", fontsize=10)
    axes[1].axis('off')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical',
                        fraction=0.035, pad=0.02)
    cbar.set_label('e3d distances', fontsize=8)

    legend_txt = "  ".join(
        [f"n{n}={d:.2f}({cat})" for n, cat, d, _ in label_data]
    )
    fig.text(0.5, 0.01, legend_txt, ha='center', fontsize=7,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig

def create_side_by_side_video(ep_dir, output_path="side_by_side.mp4", fps=5):
    """
    Creates a video with RGB and costmap frames side by side.
    
    Args:
        ep_dir:      episode directory containing frame_XXX subdirs
        output_path: output .mp4 path
        fps:         frames per second
    """
    import cv2
    import glob

    frame_dirs = sorted(glob.glob(os.path.join(ep_dir, "frame_*")))
    if not frame_dirs:
        print(f"No frames found in {ep_dir}")
        return

    writer = None

    for frame_dir in frame_dirs:

        # ── Load costmap ──────────────────────────────────────
        fd_idx    = int(os.path.basename(frame_dir).split("_")[1])
        costmap   = show_frame_pathlengths_heatmap(ep_dir, fd_idx)
        costmap.canvas.draw()
        costmap_rgba = np.array(costmap.canvas.buffer_rgba())
        costmap_bgr = cv2.cvtColor(costmap_rgba, cv2.COLOR_RGBA2BGR)

        h, w = costmap_bgr.shape[:2]
        cost_resized = cv2.resize(costmap_bgr, (w, h))

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = cost_resized

        if writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        writer.write(canvas)

    if writer:
        writer.release()
        print(f"Video saved to {output_path}")
    else:
        print("No frames written.")