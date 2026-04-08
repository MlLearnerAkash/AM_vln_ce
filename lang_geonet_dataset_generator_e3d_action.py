"""
lang_geonet_dataset_generator_e3d_action.py
---------------------------------------------
Online dataset generator for LangGeoNet using E3D (Euclidean 3-D) inter-object
distances as frame-level labels, **with per-frame next-action instructions**.

Extends lang_geonet_dataset_generator_e3d.py by adding a "peek" at the
immediately following action at every timestep.  The peeked action is mapped
to a natural-language instruction and stored as `next_action_instruction`
alongside every frame.

Action → instruction mapping
-----------------------------
    STOP         (0 / None) → "Stop here"
    MOVE_FORWARD (1)        → "Move forward"
    TURN_LEFT    (2)        → "Turn left"
    TURN_RIGHT   (3)        → "Turn right"

HDF5 layout (per frame)
-----------------------
    /<episode_id>/frames/<idx>/rgb                    (H x W x 3  uint8)
    /<episode_id>/frames/<idx>/frame_idx              (scalar int)
    /<episode_id>/frames/<idx>/next_action_instruction (scalar string)
"""

import io
import os
import sys
import pickle

import h5py
import numpy as np
import networkx as nx
import random
import tqdm
from PIL import Image

import utils_sim_traj as ust
from utils_sim import *
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image

from utils.e3d_costmap_visualizer import show_frame_pathlengths_heatmap, create_side_by_side_video

_ORN_LIBS = "/data/ws/object-rel-nav"
if _ORN_LIBS not in sys.path:
    sys.path.insert(0, _ORN_LIBS)

from libs.common.utils import get_instance_id_to_all_dict, mask_to_rle_numpy

EXCLUDE_CATS = ["ceiling", "beam", "objects", "lighting", "column", "misc", "railing", "floor"]

# ---------------------------------------------------------------------------
# Action → natural-language instruction (contextual with paraphrase pool)
# ---------------------------------------------------------------------------

# Habitat action indices (HabitatSimActions)
#   STOP         = 0
#   MOVE_FORWARD = 1
#   TURN_LEFT    = 2
#   TURN_RIGHT   = 3
# get_next_action() returns None when the agent has reached the goal (→ STOP).
#
# Templates containing {obj} are used when an anchor object is available.
# Templates without {obj} are always eligible as fallback.

STOP_INSTRUCTION = "Stop here"

ACTION_TEMPLATES = {
    0: [  # STOP
        "Stop here",
        "You've arrived",
        "This is the destination",
        "Stop near the {obj}",
        "Halt by the {obj}",
        "Your destination is near the {obj}",
    ],
    1: [  # MOVE_FORWARD
        "Move forward",
        "Go straight ahead",
        "Continue forward",
        "Head toward the {obj}",
        "Walk forward past the {obj}",
        "Proceed ahead toward the {obj}",
        "Keep going straight toward the {obj}",
    ],
    2: [  # TURN_LEFT
        "Turn left",
        "Rotate left",
        "Bear left",
        "Turn left toward the {obj}",
        "Go left near the {obj}",
        "Swing left past the {obj}",
    ],
    3: [  # TURN_RIGHT
        "Turn right",
        "Rotate right",
        "Bear right",
        "Turn right toward the {obj}",
        "Go right near the {obj}",
        "Swing right past the {obj}",
    ],
}


def _compute_e3d_norm(
    mask_dicts: list,
    instances: list,
    instance_id_dict: dict,
) -> np.ndarray:
    """
    Per-instance mean pairwise Euclidean distance to all co-visible instances,
    min-max normalised to [0, 1].  Low value → scene-central object.

    Returns
    -------
    e3d_norm : np.ndarray [K] float32  (zeros when K <= 1)
    """
    K = len(mask_dicts)
    if K <= 1:
        return np.zeros(K, dtype=np.float32)
    centers = np.array(
        [instance_id_dict.get(iid, {}).get('obb_center', np.zeros(3)) for iid in instances],
        dtype=np.float32,
    )  # [K, 3]
    diff     = centers[:, None, :] - centers[None, :, :]   # [K, K, 3]
    pairwise = np.linalg.norm(diff, axis=-1)                # [K, K]
    np.fill_diagonal(pairwise, 0.0)
    raw_e3d  = pairwise.sum(axis=1) / (K - 1)              # [K]
    lo, hi   = float(raw_e3d.min()), float(raw_e3d.max())
    denom    = hi - lo if hi > lo else 1.0
    return ((raw_e3d - lo) / denom).astype(np.float32)


def _pick_anchor_candidates(
    mask_dicts: list,
    instances: list,
    action,
    width: int,
    e3d_norm: np.ndarray,
) -> list:
    """
    Build a pool of candidate anchor object names from two sources:

    1. **E3D pool** – top-3 instances with the lowest e3d_norm (most
       scene-central); "unknown" names are skipped.
    2. **Salient object** – largest-area object whose pixel centroid lies in
       the image region matching the action direction:
           MOVE_FORWARD → centre third  (width/3 ≤ x ≤ 2·width/3)
           TURN_LEFT    → left third    (x < width/3)
           TURN_RIGHT   → right third   (x > 2·width/3)
           STOP         → anywhere

    Returns a deduplicated list of category-name strings (may be empty).
    """
    K = len(mask_dicts)
    if K == 0:
        return []
    action_int = 0 if action is None else int(action)

    # ── 1. E3D pool: top-3 least-cost ────────────────────────────────────────
    order    = np.argsort(e3d_norm)   # ascending: lowest e3d first
    e3d_pool = []
    for idx in order[:3]:
        name = str(mask_dicts[idx].get('category_name', 'unknown')).lower()
        if name and name != 'unknown':
            e3d_pool.append(mask_dicts[idx]['category_name'])

    # ── 2. Salient object in directional image region ─────────────────────────
    directional = None
    best_area   = -1
    for md in mask_dicts:
        cx   = md.get('coords', [width // 2, 0])[0]
        area = md.get('area', 0)
        name = str(md.get('category_name', 'unknown')).lower()
        if name == 'unknown':
            continue
        if action_int == 1:    # MOVE_FORWARD → centre third
            in_region = (width / 3) <= cx <= (2 * width / 3)
        elif action_int == 2:  # TURN_LEFT → left third
            in_region = cx < (width / 3)
        elif action_int == 3:  # TURN_RIGHT → right third
            in_region = cx > (2 * width / 3)
        else:                  # STOP → anywhere
            in_region = True
        if in_region and area > best_area:
            best_area   = area
            directional = md['category_name']

    # ── deduplicate while preserving insertion order ──────────────────────────
    seen, pool = set(), []
    for name in e3d_pool + ([directional] if directional else []):
        if name not in seen:
            seen.add(name)
            pool.append(name)
    return pool


def _action_to_instruction_contextual(
    action,
    mask_dicts: list,
    instances: list,
    width: int,
    e3d_norm: np.ndarray,
) -> str:
    """
    Sample a varied natural-language instruction for `action`.

    An anchor object is chosen at random from the candidate pool
    (top-3 least e3d-cost + directionally salient object).  If the pool is
    empty, only plain (object-free) templates are used.
    """
    action_int = 0 if action is None else int(action)
    templates  = ACTION_TEMPLATES.get(action_int, ACTION_TEMPLATES[0])
    candidates = _pick_anchor_candidates(mask_dicts, instances, action, width, e3d_norm)
    if candidates:
        anchor = random.choice(candidates)
        return random.choice(templates).format(obj=anchor)
    plain = [t for t in templates if '{obj}' not in t]
    return random.choice(plain) if plain else templates[0]


# ---------------------------------------------------------------------------
# Scene helpers (unchanged from e3d generator)
# ---------------------------------------------------------------------------

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
    areaThresh = np.ceil(0.001 * semantic.shape[0] * semantic.shape[1])
    instaIds = np.unique(semantic)
    masks = semantic[None, :, :] == instaIds[:, None, None]
    maskDicts = []
    for i in range(masks.shape[0]):
        area = np.sum(masks[i])
        if area <= areaThresh:
            continue
        if filterInstaIDs is not None and instaIds[i] in filterInstaIDs:
            continue
        maskDicts.append({
            'area':          area,
            'bbox':          cv2.boundingRect(masks[i].astype(np.uint8)),
            'instance_id':   instaIds[i],
            'category_id':   instaIdx2catIdx[instaIds[i], 1] if instaIdx2catIdx is not None else None,
            'category_name': str(instaIdx2catName[instaIds[i], 1]) if instaIdx2catName is not None else None,
            'segmentation':  masks[i],
            'coords':        np.array(np.nonzero(masks[i])).mean(1)[::-1].astype(int),
        })
    return maskDicts


def _get_insta_DA_edges(dMat, max_temporal_gap, nodeID_to_imgRegionIdx):
    """Build data-association edges from a boolean instance-match matrix."""
    N = dMat.shape[0]
    da_edges = []
    seen = set()
    for i in range(N):
        for j in range(i + 1, N):
            if not dMat[i, j]:
                continue
            fi = nodeID_to_imgRegionIdx[i, 0]
            fj = nodeID_to_imgRegionIdx[j, 0]
            if fi == fj:
                continue
            if abs(fi - fj) > max_temporal_gap:
                continue
            key = (i, j)
            if key not in seen:
                seen.add(key)
                da_edges.append(key)
    return da_edges, None


# ---------------------------------------------------------------------------
# HDF5 writer (extended to store next_action_instruction per frame)
# ---------------------------------------------------------------------------

def save_episode_to_hdf5_with_actions(hdf5_path: str, ep_data: dict) -> None:
    """
    Append one episode into an HDF5 file, including per-frame mask arrays
    and next-action labels.

    HDF5 layout
    -----------
    /<episode_id>/
        instruction          (scalar string — navigation instruction)
        graph                (raw bytes  — pickled networkx graph)
        frames/
            <000>/
                rgb                       (H x W x 3  uint8,  gzip-4)
                masks                     (K x H x W  uint8,  gzip-4)
                class_ids                 (K          int64)
                instance_ids              (K          int64)
                e3d_distances             (K          float32)  normalised [0,1]
                frame_idx                 (scalar int)
                next_action               (scalar int  0-3)
                next_action_instruction   (scalar string)
            <001>/
                ...
    """
    episode_id = str(ep_data["episode_id"])

    with h5py.File(hdf5_path, "a") as hf:
        if episode_id in hf:
            del hf[episode_id]

        ep_grp = hf.create_group(episode_id)

        ep_grp.create_dataset("instruction", data=ep_data["instruction"])

        buf = io.BytesIO()
        pickle.dump(ep_data["graph"], buf)
        graph_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        ep_grp.create_dataset(
            "graph",
            data=graph_bytes,
            compression="gzip",
            compression_opts=4,
        )

        frames_grp = ep_grp.create_group("frames")
        for fd in ep_data["frame_data"]:
            frame_key = f"{fd['frame_idx']:03d}"
            frame_grp = frames_grp.create_group(frame_key)

            # ── RGB ──────────────────────────────────────────────────────────
            rgb = np.asarray(fd["rgb"], dtype=np.uint8)
            H, W = rgb.shape[:2]
            frame_grp.create_dataset(
                "rgb", data=rgb, compression="gzip", compression_opts=4
            )

            # ── K-channel mask arrays (mirrors _save_frame_data_e3d) ─────────
            mask_dicts    = fd.get("mask_dicts", [])
            K             = len(mask_dicts)
            if K > 0:
                masks = np.stack(
                    [m["segmentation"].astype(np.uint8) for m in mask_dicts], axis=0
                )  # [K, H, W]
                class_ids    = np.array(
                    [m["category_id"] if m["category_id"] is not None else -1
                     for m in mask_dicts], dtype=np.int64
                )
                instance_ids = np.array(
                    [m["instance_id"] for m in mask_dicts], dtype=np.int64
                )
                e3d_distances = np.asarray(
                    fd.get("e3d_norm", np.zeros(K, dtype=np.float32)), dtype=np.float32
                )
            else:
                masks         = np.zeros((0, H, W), dtype=np.uint8)
                class_ids     = np.zeros(0, dtype=np.int64)
                instance_ids  = np.zeros(0, dtype=np.int64)
                e3d_distances = np.zeros(0, dtype=np.float32)

            frame_grp.create_dataset(
                "masks", data=masks, compression="gzip", compression_opts=4
            )
            frame_grp.create_dataset("class_ids",     data=class_ids)
            frame_grp.create_dataset("instance_ids",  data=instance_ids)
            frame_grp.create_dataset("e3d_distances", data=e3d_distances)

            # ── scalars / labels ─────────────────────────────────────────────
            frame_grp.create_dataset("frame_idx",   data=fd["frame_idx"])
            frame_grp.create_dataset("next_action", data=fd.get("next_action", 0))
            frame_grp.create_dataset(
                "next_action_instruction",
                data=fd.get("next_action_instruction", STOP_INSTRUCTION),
            )


def load_episode_from_hdf5_with_actions(hdf5_path: str, episode_id: str) -> dict:
    """
    Read one episode back from the HDF5 file.

    Returns a dict with keys:
        episode_id              : str
        instruction             : str  (navigation instruction)
        graph                   : original Python object (unpickled)
        frame_data              : list of dicts with keys
                                      frame_idx, rgb, next_action_instruction
    """
    episode_id = str(episode_id)

    with h5py.File(hdf5_path, "r") as hf:
        if episode_id not in hf:
            raise KeyError(f"Episode '{episode_id}' not found in {hdf5_path}")

        ep_grp = hf[episode_id]

        raw_instr = ep_grp["instruction"][()]
        instruction = raw_instr.decode("utf-8") if isinstance(raw_instr, bytes) else str(raw_instr)

        graph_bytes = ep_grp["graph"][()].tobytes()
        graph = pickle.loads(graph_bytes)

        frame_data = []
        frames_grp = ep_grp["frames"]
        for frame_key in sorted(frames_grp.keys()):
            fg = frames_grp[frame_key]
            raw_nai = fg["next_action_instruction"][()]
            nai = raw_nai.decode("utf-8") if isinstance(raw_nai, bytes) else str(raw_nai)
            frame_data.append({
                "frame_idx":               int(fg["frame_idx"][()]),
                "rgb":                     fg["rgb"][()],
                "masks":                   fg["masks"][()] if "masks" in fg else None,
                "class_ids":               fg["class_ids"][()] if "class_ids" in fg else None,
                "instance_ids":            fg["instance_ids"][()] if "instance_ids" in fg else None,
                "e3d_distances":           fg["e3d_distances"][()] if "e3d_distances" in fg else None,
                "next_action":             int(fg["next_action"][()]) if "next_action" in fg else 0,
                "next_action_instruction": nai,
            })

    return {
        "episode_id":  episode_id,
        "instruction": instruction,
        "graph":       graph,
        "frame_data":  frame_data,
    }


# ---------------------------------------------------------------------------
# Episode generator with next-action peek
# ---------------------------------------------------------------------------

def episode_generator(env, data_root, num_episodes: int = 10):
    """
    Drive the reference-path follower through `num_episodes` episodes.

    At every step the generator "peeks" at the immediately following action
    by querying the follower again after taking the current step.  The peeked
    action is translated to a natural-language instruction and stored in
    frame_data under the key ``next_action_instruction``.

    Yields
    ------
    dict with keys:
        episode_id             : str
        instruction            : str   (navigation instruction)
        graph                  : nx.Graph
        frame_data             : list of per-frame dicts, each containing
                                     frame_idx, rgb, mask_dicts,
                                     instance_id_dict,
                                     next_action_instruction
    """
    from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
    from habitat_sim.utils.common import quat_from_magnum
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

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

            agent_config = agent.agent_config
            sensor_spec  = agent_config.sensor_specifications[0]
            hfov   = float(sensor_spec.hfov)
            width  = int(sensor_spec.resolution[1])
            height = int(sensor_spec.resolution[0])
            K_mat  = getK_fromParams(hfov, width, height)

            # ── graph initialisation ─────────────────────────────────────────
            G             = nx.Graph()
            numNodesCurr  = 0
            temporalEdges = []
            frame_data    = []
            frame_idx     = 0
            video_frames  = []

            num_waypoints = len(reference_path)

            for wp_idx, point in enumerate(reference_path):
                # Pre-compute next waypoint for peek fall-through
                next_point = reference_path[wp_idx + 1] if wp_idx + 1 < num_waypoints else None

                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    step_result = env.step(best_action)
                    if isinstance(step_result, tuple):
                        obs, _, done, info = step_result
                    else:
                        obs = step_result

                    # ── peek: what is the next action from the new state? ────
                    # Query the follower again from the updated agent position.
                    peek_action = follower.get_next_action(point)
                    if peek_action is None:
                        # Reached this waypoint; peek at the following one.
                        if next_point is not None:
                            peek_action = follower.get_next_action(next_point)
                        # If peek_action is still None (or no next waypoint),
                        # the episode is about to end → STOP.
                    # (instruction generated after mask_dicts/instances are finalised)

                    rgb      = obs["rgb"]
                    depth    = obs["depth"]
                    semantic = obs["semantic"]

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

                    assert len(mask_dicts) == len(instances)

                    mask_dicts_aligned = []
                    filtered_instances = []
                    for inst_id in instances:
                        matched = [m for m in mask_dicts if m['instance_id'] == inst_id]
                        if matched:
                            md = matched[0]
                            if md.get('category_name', '').lower() in EXCLUDE_CATS:
                                continue
                            mask_dicts_aligned.append(md)
                            filtered_instances.append(inst_id)
                        else:
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
                    instances  = filtered_instances

                    # ── contextual next-action instruction ───────────────────
                    e3d_norm        = _compute_e3d_norm(mask_dicts, instances, instance_id_dict)
                    peek_action_int = 0 if peek_action is None else int(peek_action)
                    next_action_instruction = _action_to_instruction_contextual(
                        peek_action, mask_dicts, instances, width, e3d_norm
                    )

                    numMasks = len(mask_dicts)
                    nodes, edges = [], []

                    for j in range(numMasks):
                        md     = mask_dicts[j]
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
                        'frame_idx':               frame_idx,
                        'rgb':                     rgb,
                        'mask_dicts':              mask_dicts,
                        'instance_id_dict':        instance_id_dict,
                        'e3d_norm':                e3d_norm,          # [K] float32, already computed
                        'next_action':             peek_action_int,
                        'next_action_instruction': next_action_instruction,
                    })

                    vis_frame = visualize_segmentation_with_e3d(
                        rgb              = rgb,
                        mask_dicts       = mask_dicts,
                        instances        = instances,
                        pls_intra        = pls_intra,
                        instance_id_dict = instance_id_dict,
                        alpha            = 0.40,
                        show_e3d         = True,
                        show_instance_ids= True,
                    )
                    obs["vis_img"] = vis_frame
                    video_frame = observations_to_image(obs, info)
                    video_frame = append_text_to_image(
                        video_frame,
                        f"{instruction}  |  next: {next_action_instruction}",
                    )
                    if video_frames:
                        th, tw = video_frames[0].shape[:2]
                        if video_frame.shape[0] != th or video_frame.shape[1] != tw:
                            video_frame = cv2.resize(video_frame, (tw, th))
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
                video_option="disk",
                video_dir=video_path,
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
                dMat  = instaIds[:, None] == instaIds[None, :]
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

            print("  precomputing all-pairs shortest paths lengths...", end="")
            all_paths_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='e3d'))
            all_paths_lengths = np.array([
                [all_paths_lengths[src].get(tgt, 1e6) for tgt in G.nodes()]
                for src in G.nodes()
            ])
            all_paths_lengths = np.nan_to_num(all_paths_lengths, nan=1e6, posinf=1e6, neginf=1e6)
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
    Collect episodes from the Habitat simulator and write the full e3d+action
    dataset.

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
    HDF5_PATH = os.path.join(data_root, f"e3d_action_{split}_ep{num_episodes}.h5")

    try:
        with open(split_file, "w") as fh_split:
            for ep_data in episode_generator(env, data_root, num_episodes=num_episodes):
                episode_id = ep_data["episode_id"]
                ep_dir = os.path.join(data_root, f"episode_{episode_id}")
                os.makedirs(ep_dir, exist_ok=True)

                with open(os.path.join(ep_dir, "instruction.txt"), "w") as fh:
                    fh.write(ep_data["instruction"])

                fh_split.write(f"{episode_id}\n")
                print(
                    f"  episode {ep_data['episode_id']}  "
                    f"{len(ep_data['frame_data'])} frames  → {HDF5_PATH}"
                )
                save_episode_to_hdf5_with_actions(HDF5_PATH, ep_data)

    finally:
        env.close()

    print(f"\nDataset written to {data_root}  ({split} split: {split_file})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build LangGeoNet E3D+Action dataset (online mode)"
    )
    parser.add_argument(
        "--config",
        default="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml",
        help="Path to Habitat YAML config",
    )
    parser.add_argument(
        "--data_root",
        default="/media/opervu-user/Data2/ws/data_langgeonet_e3d_action",
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
