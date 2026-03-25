"""
lang_geonet_dataset_generator_e3d.py
-------------------------------------
Online dataset generator for LangGeoNet using E3D (Euclidean 3-D) inter-object
distances as frame-level labels.

Core pipeline (mirrors create_sim_graph_topometric):
  For every episode the reference-path follower drives the agent step by step.
  At each step:
    • semantic + depth observations are obtained from the live simulator.
    • Visible object instances are detected via getMasksDictFromSemSensor.
    • Navigable floor-points near each instance are found and their intra-frame
      pairwise geodesic distances are computed (pls_intra).
    • Nodes (one per visible instance) and intra-frame edges (weighted by both
      geodesic and e3d distances) are added to a NetworkX graph.
    • Per-frame k-channel outputs are written:
          masks.npy            [K, H, W]  uint8   binary instance masks
          class_ids.npy        [K]        int64   semantic instance IDs
          e3d_distances.npy    [K]        float32 normalised mean e3d to all other
                                                  co-visible instances [0,1]
          instance_ids.npy     [K]        int64   habitat instance IDs (same order)
          rgb.png                                 RGB frame
  After all frames, DA (data-association / loop-closure) edges are added and
  the graph is saved:
          episode_graph.pickle  nx.Graph  full topometric graph for the episode

Dataset layout
--------------
data_root/
    train.txt                episode IDs, one per line
    episode_<id>/
        instruction.txt
        episode_graph.pickle  nx.Graph
        frame_000/
            rgb.png
            masks.npy          [K, H, W] uint8
            class_ids.npy      [K]       int64
            instance_ids.npy   [K]       int64
            e3d_distances.npy  [K]       float32 normalised to [0,1]
        frame_001/
            ...

E3D distance for instance j at frame i:
    raw_e3d[j] = mean over k≠j of ||obb_center_j - obb_center_k||
    (mean over all co-visible instances k in the same frame)
    normalised to [0,1] over the K instances.
"""

import os
import sys
import pickle
import time

import numpy as np
import networkx as nx
import tqdm
from PIL import Image

import utils_sim_traj as ust
from utils_sim import *
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image

from utils.e3d_costmap_visualizer import show_frame_pathlengths_heatmap, create_side_by_side_video
from utils.h5_writer import save_episode_to_hdf5

_ORN_LIBS = "/data/ws/object-rel-nav"
if _ORN_LIBS not in sys.path:
    sys.path.insert(0, _ORN_LIBS)

from libs.common.utils import get_instance_id_to_all_dict, mask_to_rle_numpy

EXCLUDE_CATS = ["ceiling", "beam", "objects", "lighting", "column", "misc", "railing", "floor"] #


def _build_insta_maps(semantic_scene):
    """Return (instaIdx2catIdx, instaIdx2catName) arrays from a semantic scene."""
    rows_idx, rows_name = [], []
    for obj in semantic_scene.objects:
        try:
            iid = int(obj.id.split("_")[-1])
        except Exception:
            continue
        rows_idx.append([iid, obj.category.index()])
        rows_name.append([iid, obj.category.name()])
    instaIdx2catIdx  = np.array(rows_idx,  dtype=object)
    instaIdx2catName = np.array(rows_name, dtype=object)
    return instaIdx2catIdx, instaIdx2catName


def _get_masks_from_semantic(semantic, instaIdx2catIdx, instaIdx2catName,
                             filterInstaIDs=None):
    areaThresh= np.ceil(0.001 * semantic.shape[0] * semantic.shape[1])
    instaIds = np.unique(semantic)
    masks = semantic[None,:,:] == instaIds[:, None, None]
    maskDicts = []
    for i in range(masks.shape[0]):
        area = np.sum(masks[i])
        if area <= areaThresh:
            continue
        if filterInstaIDs is not None and instaIds[i] in filterInstaIDs:
            continue
        maskDicts.append({
            'area': area,
            'bbox': cv2.boundingRect(masks[i].astype(np.uint8)),
            'instance_id': instaIds[i],
            'category_id': instaIdx2catIdx[instaIds[i],1] if instaIdx2catIdx is not None else None,
            'category_name': str(instaIdx2catName[instaIds[i],1]) if instaIdx2catName is not None else None,
            'segmentation': masks[i],
            'coords': np.array(np.nonzero(masks[i])).mean(1)[::-1].astype(int)
        })
    return maskDicts


# ---------------------------------------------------------------------------
# Helper: get_insta_DA_edges
# Mirrors get_insta_DA_edges used in create_sim_graph_topometric.
# Connects nodes that share the same instance ID but appear in DIFFERENT frames
# and are within `max_temporal_gap` frames of each other.
# ---------------------------------------------------------------------------

def _get_insta_DA_edges(dMat, max_temporal_gap, nodeID_to_imgRegionIdx):
    """
    Build data-association edges from a boolean instance-match matrix.

    Parameters
    ----------
    dMat                : (N, N) bool  True iff nodes i,j share an instance_id
    max_temporal_gap    : int          max allowed |frame_i - frame_j|
    nodeID_to_imgRegionIdx : (N, 2)  [:,0] = frame index, [:,1] = region index

    Returns
    -------
    da_edges : list of (i, j) tuples
    """
    N = dMat.shape[0]
    da_edges = []
    seen = set()
    for i in range(N):
        for j in range(i + 1, N):
            if not dMat[i, j]:
                continue
            fi = nodeID_to_imgRegionIdx[i, 0]
            fj = nodeID_to_imgRegionIdx[j, 0]
            if fi == fj:           # same frame → already an intra-image edge
                continue
            if abs(fi - fj) > max_temporal_gap:
                continue
            key = (i, j)
            if key not in seen:
                seen.add(key)
                da_edges.append(key)
    return da_edges, None


# ---------------------------------------------------------------------------
# Per-frame e3d output writer
# ---------------------------------------------------------------------------

def _save_frame_data_e3d(
    rgb: np.ndarray,
    mask_dicts: list,
    instance_id_dict: dict,
    output_dir: str,
) -> tuple:
    """
    Persist per-frame k-channel e3d outputs.

    Parameters
    ----------
    rgb              : (H, W, 3) uint8
    mask_dicts       : list of mask dicts (from _get_masks_from_semantic)
    instance_id_dict : {instance_id: {obb_center: np.ndarray, ...}}
    output_dir       : directory to save outputs

    Returns
    -------
    masks         : [K, H, W] uint8
    class_ids     : [K] int64   (semantic category IDs)
    instance_ids  : [K] int64   (habitat instance IDs)
    e3d_distances : [K] float32 normalised mean e3d to co-visible instances
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── RGB ──────────────────────────────────────────────────────────────────
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(output_dir, "rgb.png"))

    K = len(mask_dicts)
    if K == 0:
        H, W = rgb.shape[:2]
        empty_masks = np.zeros((0, H, W), dtype=np.uint8)
        np.save(os.path.join(output_dir, "masks.npy"),         empty_masks)
        np.save(os.path.join(output_dir, "class_ids.npy"),     np.zeros(0, np.int64))
        np.save(os.path.join(output_dir, "instance_ids.npy"),  np.zeros(0, np.int64))
        np.save(os.path.join(output_dir, "e3d_distances.npy"), np.zeros(0, np.float32))
        return empty_masks, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)

    masks        = np.stack([m['segmentation'].astype(np.uint8) for m in mask_dicts], axis=0)
    class_ids    = np.array([m['category_id']  for m in mask_dicts], dtype=np.int64)
    instance_ids = np.array([m['instance_id']  for m in mask_dicts], dtype=np.int64)

    # ── E3D: mean pairwise inter-object Euclidean distance ──────────────────
    # Collect OBB centres for visible instances
    centers = []
    for m in mask_dicts:
        iid = m['instance_id']
        if iid in instance_id_dict:
            centers.append(np.array(instance_id_dict[iid]['obb_center'], dtype=np.float32))
        else:
            centers.append(np.zeros(3, dtype=np.float32))
    centers = np.stack(centers)   # [K, 3]

    # For each instance j, raw_e3d[j] = mean over all k≠j of ||c_j - c_k||
    raw_e3d = np.zeros(K, dtype=np.float32)
    if K > 1:
        diff = centers[:, None, :] - centers[None, :, :]          # [K, K, 3]
        pairwise = np.linalg.norm(diff, axis=-1)                   # [K, K]
        np.fill_diagonal(pairwise, 0.0)
        raw_e3d = pairwise.sum(axis=1) / (K - 1)                  # [K]

    # Normalise to [0, 1]
    e3d_min, e3d_max = float(raw_e3d.min()), float(raw_e3d.max())
    denom = e3d_max - e3d_min if e3d_max > e3d_min else 1.0
    e3d_distances = ((raw_e3d - e3d_min) / denom).astype(np.float32)

    np.save(os.path.join(output_dir, "masks.npy"),         masks)
    np.save(os.path.join(output_dir, "class_ids.npy"),     class_ids)
    np.save(os.path.join(output_dir, "instance_ids.npy"),  instance_ids)
    np.save(os.path.join(output_dir, "e3d_distances.npy"), e3d_distances)

    return masks, class_ids, instance_ids, e3d_distances


# ---------------------------------------------------------------------------
# Episode generator: drives the follower, builds the graph, yields episode data
# ---------------------------------------------------------------------------

def episode_generator(env, data_root, num_episodes: int = 10):

    from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
    from habitat_sim.utils.common import quat_from_magnum

    follower = ShortestPathFollowerCompat(
        env._env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = "geodesic_path"

    with tqdm.tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
         
        for _ in range(num_episodes):
            step_count = 0
            obs = env.reset()
            episode_id     = env.current_episode.episode_id
            reference_path = env.current_episode.reference_path
            instruction    = env.current_episode.instruction.instruction_text

            sim   = env._env.sim
            agent = sim.get_agent(0)

            # ── one-time per-episode scene metadata ──────────────────────────
            semantic_scene   = sim.semantic_annotations()
            instance_id_dict = get_instance_id_to_all_dict(
                semantic_scene, save_explicit_dict=True
            )
            instaIdx2catIdx, instaIdx2catName = _build_insta_maps(semantic_scene)

            # camera intrinsics (used by get_navigable_points_on_instances)
            agent_config = agent.agent_config
            sensor_spec  = agent_config.sensor_specifications[0]  # first sensor
            hfov   = float(sensor_spec.hfov)
            width  = int(sensor_spec.resolution[1])
            height = int(sensor_spec.resolution[0])
            K_mat  = getK_fromParams(hfov, width, height)  # from utils_sim (star import)

            # ── graph initialisation ─────────────────────────────────────────
            G= nx.Graph()
            numNodesCurr  = 0
            temporalEdges = []
            frame_data    = []
            frame_idx     = 0
            video_frames  = []
            for point in reference_path:
                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    step_result = env.step(best_action)
                    if isinstance(step_result, tuple):
                        obs, _, done, info = step_result
                    else:
                        obs = step_result

                    rgb      = obs["rgb"]                      # (H, W, 3) uint8
                    depth    = obs["depth"]                    # (H, W)    float32
                    semantic = obs["semantic"]                 # (H, W)    int32

                    curr_state = agent.get_state()

                    # ── instance masks ───────────────────────────────────────
                    mask_dicts = _get_masks_from_semantic(
                        semantic, instaIdx2catIdx, instaIdx2catName,
                        filterInstaIDs=[0]
                    )

                    # ── navigable points + intra-frame pairwise geodesics ────
                    points_data = ust.get_navigable_points_on_instances(
                        sim, K_mat, curr_state, depth, semantic,
                        numSamples=5, intra_pls=True,
                        filterByArea=True, filterInstaIDs=[0]
                    )

                    if points_data is None:
                        print(f"[warn] points_data is None at frame {frame_idx}, skipping.")
                        if done:
                            break
                        frame_idx += 1
                        continue

                    p3d_c, p3d_w, p_w_nav, instances, inds, pointsAll, pls_intra = points_data

                    assert(len(mask_dicts) == len(instances))

                    mask_dicts_aligned = []
                    filtered_instances = []
                    for inst_id in instances:
                        matched = [m for m in mask_dicts if m['instance_id'] == inst_id]
                        if matched:
                            md = matched[0]
                            # skip structural categories
                            if md.get('category_name', '').lower() in EXCLUDE_CATS:
                                continue
                            mask_dicts_aligned.append(md)
                            filtered_instances.append(inst_id)
                        else:
                            # check instance_id_dict for category name before adding blank
                            cat_name = instance_id_dict.get(inst_id, {}).get('category_name', 'unknown')
                            if cat_name.lower() in EXCLUDE_CATS:
                                continue
                            mask_blank = np.zeros((height, width), dtype=bool)
                            mask_dicts_aligned.append({
                                'segmentation':  mask_blank,
                                'instance_id':   int(inst_id),
                                'category_id':   -1,
                                'category_name': 'unknown',
                            })
                            filtered_instances.append(inst_id)
                    mask_dicts = mask_dicts_aligned
                    instances= filtered_instances

                    numMasks = len(mask_dicts)
                    nodes, edges = [], []

                    for j in range(numMasks):
                        md = mask_dicts[j]
                        inst_j = instances[j]

                        node_dict = {
                            "map":            [frame_idx, j],
                            "instance_dict":  instance_id_dict.get(inst_j, {}),
                            "agent_position": curr_state.position,
                            "agent_rotation": np.array(
                                ust.quat_to_magnum(curr_state.rotation).to_matrix()
                            ),
                            **{k: v for k, v in md.items() if k != 'segmentation'},
                            "segmentation":   mask_to_rle_numpy(
                                md['segmentation'][None, ...]
                            )[0],
                        }
                        nodes.append((numNodesCurr + j, node_dict))

                        # intra-frame edges with both geodesic and e3d weights
                        for k in range(j + 1, numMasks):
                            inst_k = instances[k]
                            c_j = np.array(instance_id_dict.get(inst_j, {}).get('obb_center', np.zeros(3)))
                            c_k = np.array(instance_id_dict.get(inst_k, {}).get('obb_center', np.zeros(3)))
                            edges.append((
                                numNodesCurr + j,
                                numNodesCurr + k,
                                {
                                    'geodesic_min': pls_intra[j, k, 0],
                                    'geodesic_avg': pls_intra[j, k, 1],
                                    'geodesic_max': pls_intra[j, k, 2],
                                    'e3d':          float(np.linalg.norm(c_j - c_k)),
                                }
                            ))

                    G.add_nodes_from(nodes)
                    G.add_edges_from(edges)

                    if frame_idx != 0:
                        temporalEdges.append(
                            (numNodesCurr - 1, numNodesCurr, {'sim': 0})
                        )
                    numNodesCurr += numMasks

                    # ── collect per-frame output ─────────────────────────────
                    frame_data.append({
                        'frame_idx':    frame_idx,
                        'rgb':          rgb,
                        'mask_dicts':   mask_dicts,
                        'instance_id_dict': instance_id_dict,
                    })

                    #visualize e3d cost
                    vis_frame = visualize_segmentation_with_e3d(
                                                                rgb          = rgb,
                                                                mask_dicts   = mask_dicts,  
                                                                instances    = instances,
                                                                pls_intra    = pls_intra,
                                                                instance_id_dict = instance_id_dict,
                                                                alpha        = 0.40,
                                                                show_e3d     = True,
                                                                show_instance_ids = True,
                                                            )
                    # combined = np.concatenate([rgb, vis_frame], axis=1)
                    obs["vis_img"] = vis_frame
                    video_frame = observations_to_image(obs, info)
                    video_frame = append_text_to_image(video_frame, instruction)
                    video_frames.append(video_frame)
                    frame_idx += 1
                    if done:
                        break
                    step_count += 1
                if env._env.episode_over:
                    break
            
            # Save episode video
            video_path = os.path.join(data_root, f"episode_{episode_id}")
            generate_video(
                    video_option="disk", #config.VIDEO_OPTION,
                    video_dir=video_path, #config.VIDEO_DIR,
                    images=video_frames,
                    episode_id=f"{episode_id}",
                    checkpoint_idx=0,
                    metrics={"steps": step_count},
                    tb_writer=None,
                )
            
            # ── DA (loop-closure) edges ──────────────────────────────────────
            if G.number_of_nodes() > 0:
                nodeID_to_imgRegionIdx = np.array(
                    [G.nodes[n]['map'] for n in G.nodes]
                )
                instaIds = np.array(
                    [G.nodes[n]['instance_id'] for n in G.nodes]
                )
                dMat = (instaIds[:, None] == instaIds[None, :])
                da_e, _ = _get_insta_DA_edges(dMat, 3, nodeID_to_imgRegionIdx)
                da_edges_with_attr = [
                    (e[0], e[1], {
                        'geodesic_min': 0.0,
                        'geodesic_avg': 0.0,
                        'geodesic_max': 0.0,
                        'e3d':          0.0,
                    }) for e in da_e
                ]
                G.add_edges_from(da_edges_with_attr)

            G.graph['temporalEdges'] = np.array(temporalEdges)
            G.graph['cfg'] = {
                'episode_id': episode_id,
                'width':      width,
                'height':     height,
                'hfov':       hfov,
            }
            precompute_allPathsLengths= True

            if precompute_allPathsLengths:
                print("  precomputing all-pairs shortest paths lengths...", end="")
                all_paths_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='e3d'))
                all_paths_lengths = np.array([[all_paths_lengths[src].get(tgt, 1e6) for tgt in G.nodes()] for src in G.nodes()])
                all_paths_lengths= np.nan_to_num(all_paths_lengths, nan=1e6, posinf=1e6, neginf=1e6)
                G.graph['all_paths_lengths'] = all_paths_lengths

            print(
                f"  episode {episode_id}: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges, "
                f"connected={nx.is_connected(G) if G.number_of_nodes() > 0 else 'N/A'}"
            )

            yield {
                'episode_id':  episode_id,
                'instruction': instruction,
                'graph':       G,
                'frame_data':  frame_data,
            }
            pbar.update()


# ---------------------------------------------------------------------------
# Online entry point
# ---------------------------------------------------------------------------

def _run_online(
    config_path: str,
    data_root: str,
    num_episodes: int,
    split: str = "train",
) -> None:
    """
    Collect episodes from the Habitat simulator and write the full e3d dataset.

    Parameters
    ----------
    config_path  : path to Habitat YAML config
    data_root    : root output directory
    num_episodes : number of episodes to collect
    split        : "train" or "val"
    """
    from vlnce_baselines.common.environments import VLNCEDaggerEnv
    from vlnce_baselines.config.default import get_config

    config = get_config(config_paths=config_path)
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["TOP_DOWN_MAP_VLNCE"]
    config.freeze()

    os.makedirs(data_root, exist_ok=True)
    split_file = os.path.join(data_root, f"{split}.txt")

    env = VLNCEDaggerEnv(config=config)
    HDF5_PATH = os.path.join(data_root, f"{split}.h5")

    try:
        with open(split_file, "w") as fh_split:
            for ep_data in episode_generator(env, data_root, num_episodes=num_episodes):
                episode_id = ep_data["episode_id"]
                ep_dir = os.path.join(data_root, f"episode_{episode_id}")
                os.makedirs(ep_dir, exist_ok=True)

                # instruction.txt
                with open(os.path.join(ep_dir, "instruction.txt"), "w") as fh:
                    fh.write(ep_data["instruction"])

                # episode graph
                # graph_path = os.path.join(ep_dir, "episode_graph.pickle")
                # with open(graph_path, "wb") as fh:
                #     pickle.dump(ep_data["graph"], fh)

                # per-frame files
                # for fd in ep_data["frame_data"]:
                #     frame_dir = os.path.join(ep_dir, f"frame_{fd['frame_idx']:03d}")
                #     _save_frame_data_e3d(
                #         rgb=fd["rgb"],
                #         mask_dicts=fd["mask_dicts"],
                #         instance_id_dict=fd["instance_id_dict"],
                #         output_dir=frame_dir,
                #     )

                #NOTE: visualize costmap
                # video_path = os.path.join(ep_dir, "side_by_side.mp4")
                # create_side_by_side_video(ep_dir, output_path=video_path, fps=5)

                fh_split.write(f"{episode_id}\n")
                # print(
                #     f"  episode {episode_id}  {len(ep_data['frame_data'])} frames"
                #     f"  graph saved → {graph_path}"
                # )
                print(
                    f"  episode {ep_data['episode_id']}  "
                    f"{len(ep_data['frame_data'])} frames  → {HDF5_PATH}"
                )
                save_episode_to_hdf5(HDF5_PATH, ep_data)

    finally:
        env.close()

    print(f"\nDataset written to {data_root}  ({split} split: {split_file})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build LangGeoNet E3D dataset (online mode only)"
    )
    parser.add_argument(
        "--config",
        default="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml",
        help="Path to Habitat YAML config",
    )
    parser.add_argument(
        "--data_root",
        default="/media/opervu-user/Data2/ws/data_langgeonet_e3d",
        help="Root output directory for the dataset",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=500,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="Dataset split (writes train.txt or val.txt)",
    )

    args = parser.parse_args()
    _run_online(
        config_path=args.config,
        data_root=args.data_root,
        num_episodes=args.num_episodes,
        split=args.split,
    )
