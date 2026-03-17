"""
build_langgeonet_dataset.py

Collects Habitat navigation episodes (following the reference path), runs
LangGeoNet on every frame to obtain per-object path-length costs, and writes
the training dataset expected by the costmap predictor:

  <out_dir>/
    <traj_name>/
      images/
        00000.png
        00001.png
        ...
      traj_data.pkl   {position: (N,2) float32,  yaw: (N,) float64}
    train/
      traj_names.txt  one trajectory name per line
    test/
      traj_names.txt  one trajectory name per line
  <out_dir>/costmaps.h5

costmaps.h5 key layout  (one group per node)
  <traj_name>_<node_idx>/
      size          attribute  [H, W]
      img_masks/
          0         (M,)  int64  flat pixel indices for object 0
          1         ...
      img_pls       (K,)  float64  path-length cost per object

LangGeoNet predicts normalised geodesic distances [0,1] for each object
mask; these are scaled to [0, 200] to match the cost range used during
training (same convention as get_goal_image_langgeonet in eval_obj_react.py).

Usage
-----
  python build_langgeonet_dataset.py \\
      --out_dir ./data/my_dataset \\
      --num_episodes 100 \\
      --langgeonet_ckpt ./costmap_predictor/langgeonet/checkpoints/best_model.pt \\
      --config ./vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \\
      --image_size 120 160 \\
      --train_ratio 0.8

The script resizes every output image (and the semantic map used to derive
object masks) to --image_size so that the images/ folder and the .h5 file
use a consistent resolution.
"""

import argparse
import math
import os
import pickle
import random

import cv2
import h5py
import numpy as np
import tqdm
from PIL import Image

from costmap_predictor.langgeonet.inference import LangGeoNetPredictor
from dataset.episode_generator import episode_generator
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances
from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.config.default import get_config


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw(rotation) -> float:
    theta_habitat = 2.0 * math.atan2(float(rotation.y), float(rotation.w))
    # Habitat Y-rotation → standard math yaw (forward = [cos φ, sin φ])
    # In Habitat (x,z): forward = [-sin θ, -cos θ], so φ = -(θ + π/2)
    return -(theta_habitat + math.pi / 2)


def _position_to_xy(position) -> np.ndarray:
    """Return world-frame (x, z) in metres from a 3-D Habitat position."""
    return np.array([float(position[0]), float(position[2])], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Mask builder
# ─────────────────────────────────────────────────────────────────────────────

def _gt_costs_from_distances(
    object_ids: list,
    distances_dict: dict,
    cost_scale: float,
) -> np.ndarray:
    """
    Convert a {object_id: geodesic_distance} dict to a per-object cost array
    that matches the ordering of *object_ids* (as returned by
    ``_masks_from_semantic``).

    Distances that are ``None`` or ``inf`` are clamped to 55 m (same cap used
    in ``get_object_geodesic_distances``).  The values are then normalised
    per-frame to [0, 1] and scaled by *cost_scale* so that they are on the
    same range as the LangGeoNet outputs.
    """
    raw = np.array(
        [float(distances_dict.get(oid) or 55.0) for oid in object_ids],
        dtype=np.float64,
    )
    mn, mx = raw.min(), raw.max()
    norm = (raw - mn) / (mx - mn) if mx > mn else np.zeros_like(raw)
    return (norm * cost_scale).astype(np.float64)


def _masks_from_semantic(semantic: np.ndarray):
    """
    Build one binary mask per unique object (background / 0 excluded).

    Parameters
    ----------
    semantic : (H, W) int32   instance-segmentation map

    Returns
    -------
    masks      : (K, H, W) bool   — may be (0, H, W) when no objects found
    object_ids : list[int]  length K
    """
    unique = [int(u) for u in np.unique(semantic) if u > 0]
    if not unique:
        H, W = semantic.shape
        return np.zeros((0, H, W), dtype=bool), []
    masks = np.stack([(semantic == uid) for uid in unique], axis=0)
    return masks, unique


def _mask_to_rle_counts(mask: np.ndarray) -> np.ndarray:
    """
    Encode a boolean (H, W) mask as uncompressed RLE counts in column-major
    order, compatible with ``rle_to_mask()`` in the dataloader.

    ``rle_to_mask`` reconstructs via::

        flat = decode(counts)       # column-major (Fortran) order
        mask = flat.reshape(W, H).T  # back to (H, W)

    So we encode by flattening mask.T in C order.
    """
    flat = mask.T.flatten()   # column-major traversal of (H, W) mask
    counts = []
    parity = False
    run = 0
    for val in flat:
        if bool(val) == parity:
            run += 1
        else:
            counts.append(run)
            parity = not parity
            run = 1
    counts.append(run)
    return np.array(counts, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dataset writer
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuilder:
    """
    Writes one trajectory at a time to the dataset directory structure.

    Directory layout
    ----------------
    out_dir/
      <traj_name>/
        images/  00000.png …
        traj_data.pkl
    out_dir/costmaps.h5 (appended incrementally)
    """

    def __init__(self, out_dir: str, h5_path: str):
        self.out_dir = out_dir
        self.h5_path = h5_path

    # ------------------------------------------------------------------
    def save_trajectory(
        self,
        traj_name: str,
        images,        # list[np.ndarray (H, W, 3) uint8]
        positions,     # list[np.ndarray (2,) float64]  world (x, z) metres
        yaws,          # list[float]  radians
        actions,       # list[int]  discrete GT action taken at each step
        masks_list,    # list[np.ndarray (K_i, H, W) bool]
        costs_list,    # list[np.ndarray (K_i,) float64]  path-length costs
    ):
        """Flush one trajectory to disk."""
        traj_dir   = os.path.join(self.out_dir, traj_name)
        images_dir = os.path.join(traj_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        N = len(images)

        # ── 1. Save PNG frames ──────────────────────────────────────────────
        for idx, img in enumerate(images):
            save_path = os.path.join(images_dir, f"{idx:05d}.png")
            img_arr = img if isinstance(img, np.ndarray) else np.array(img)
            Image.fromarray(img_arr.astype(np.uint8)).save(save_path)

        # ── 2. Save traj_data.pkl ───────────────────────────────────────────
        traj_data = {
            "position": np.stack(positions).astype(np.float32),  # (N, 2)
            "yaw":      np.array(yaws, dtype=np.float64),         # (N,)
            "action":   np.array(actions, dtype=np.int32),        # (N,)
        }
        with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        # ── 3. Append nodes to the HDF5 file ───────────────────────────────
        with h5py.File(self.h5_path, "a") as h5:
            for node_idx, (img, masks, costs) in enumerate(
                zip(images, masks_list, costs_list)
            ):
                key = f"{traj_name}_{node_idx}"
                grp = h5.require_group(key)

                img_arr = img if isinstance(img, np.ndarray) else np.array(img)
                H, W = img_arr.shape[:2]
                # store size as a dataset (dataloader reads it as key_data["size"][()])
                if "size" in grp:
                    del grp["size"]
                grp.create_dataset("size", data=np.array([H, W], dtype=np.int64))

                masks_grp = grp.require_group("img_masks")
                K = masks.shape[0]
                for k in range(K):
                    # store uncompressed RLE counts, compatible with rle_to_mask()
                    rle_counts = _mask_to_rle_counts(masks[k])
                    ds_name = str(k)
                    if ds_name in masks_grp:
                        del masks_grp[ds_name]
                    masks_grp.create_dataset(ds_name, data=rle_counts,
                                             compression="gzip")

                if "img_pls" in grp:
                    del grp["img_pls"]
                grp.create_dataset("img_pls",
                                   data=costs.astype(np.float64))

        print(f"  Saved '{traj_name}': {N} nodes → {traj_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Episode collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_and_save(
    env,
    builder: DatasetBuilder,
    num_episodes: int,
    image_size: tuple,                      # (H, W) output resolution
    cost_scale: float = 100.0,              # map [0,1] distance → [0, cost_scale]
    predictor: "LangGeoNetPredictor | None" = None,
    use_gt_geodesic: bool = False,
):
    """
    Follow the reference path for each episode, run LangGeoNet (or use GT
    geodesic distances) per frame, and write the dataset.

    Exactly one of *predictor* or *use_gt_geodesic=True* must be provided.

    Returns
    -------
    traj_names : list[str]  names of all saved trajectories, in order
    """
    if use_gt_geodesic and predictor is not None:
        raise ValueError("Specify either --use_gt_geodesic or a LangGeoNet checkpoint, not both.")
    if not use_gt_geodesic and predictor is None:
        raise ValueError("Provide a LangGeoNet checkpoint via --langgeonet_ckpt, or pass --use_gt_geodesic.")
    follower = ShortestPathFollowerCompat(
        env._env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = "geodesic_path"

    target_h, target_w = image_size
    traj_names = []

    with tqdm.tqdm(total=num_episodes, desc="Episodes") as pbar:
        episode_idx = 0
        while episode_idx < num_episodes:
            obs = env.reset()
            episode_id     = env.current_episode.episode_id
            reference_path = env.current_episode.reference_path
            instruction    = env.current_episode.instruction.instruction_text
            traj_name      = f"ep_{episode_id}"

            images, positions, yaws, actions, masks_list, costs_list = [], [], [], [], [], []
            # ── Walk the reference path ──────────────────────────────────────
            for waypoint in reference_path:
                while not env._env.episode_over:
                    action = follower.get_next_action(waypoint)
                    if action is None:
                        break

                    obs, _, done, _ = env.step(action)
                    actions.append(int(action))

                    # ── RGB frame ────────────────────────────────────────────
                    rgb = obs["rgb"]  # (H_cam, W_cam, 3) uint8
                    if rgb.shape[:2] != (target_h, target_w):
                        rgb = cv2.resize(rgb, (target_w, target_h),
                                         interpolation=cv2.INTER_LINEAR)

                    # ── Semantic map → masks ─────────────────────────────────
                    sem = obs["semantic"].astype(np.int32)   # (H_cam, W_cam)
                    if sem.shape != (target_h, target_w):
                        sem = cv2.resize(sem.astype(np.float32),
                                         (target_w, target_h),
                                         interpolation=cv2.INTER_NEAREST
                                         ).astype(np.int32)
                    masks, object_ids = _masks_from_semantic(sem)  # (K, H, W) bool
                    masks_f  = masks.astype(np.float32)             # predict_frame wants float

                    # ── Compute per-object costs ─────────────────────────────
                    if masks_f.shape[0] > 0:
                        if use_gt_geodesic:
                            gt_distances = get_object_geodesic_distances(env, sem)
                            costs = _gt_costs_from_distances(
                                object_ids, gt_distances, cost_scale
                            )
                        else:
                            distances, _, _ = predictor.predict_frame(
                                rgb, masks_f, instruction
                            )                               # (K,) in [0, 1]
                            costs = (distances * cost_scale).astype(np.float64)
                    else:
                        costs = np.array([], dtype=np.float64)

                    # ── Agent pose ────────────────────────────────────────────
                    ag_state = env._env.sim.get_agent_state()
                    pos2d    = _position_to_xy(ag_state.position)
                    yaw      = _quat_to_yaw(ag_state.rotation)

                    images.append(rgb)
                    positions.append(pos2d)
                    yaws.append(yaw)
                    masks_list.append(masks)
                    costs_list.append(costs)

                    if done:
                        break

                if env._env.episode_over:
                    break

            if not images:
                print(f"  No frames collected for episode {episode_id}, skipping.")
                continue

            builder.save_trajectory(
                traj_name, images, positions, yaws, actions, masks_list, costs_list
            )
            traj_names.append(traj_name)
            episode_idx += 1
            pbar.update()

    return traj_names


# ─────────────────────────────────────────────────────────────────────────────
# Split-file writer
# ─────────────────────────────────────────────────────────────────────────────

def write_split_files(out_dir: str, traj_names: list, train_ratio: float, seed: int = 42):
    """
    Write  <out_dir>/train/traj_names.txt  and  <out_dir>/test/traj_names.txt.
    """
    rng = random.Random(seed)
    shuffled = list(traj_names)
    rng.shuffle(shuffled)

    n_train = max(1, round(len(shuffled) * train_ratio))
    train_names = shuffled[:n_train]
    test_names  = shuffled[n_train:]

    for split, names in [("train", train_names), ("test", test_names)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "traj_names.txt"), "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"  {split}: {len(names)} trajectories → {split_dir}/traj_names.txt")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build LangGeoNet training dataset from Habitat episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir",
                   default= "/data/ws/VLN-CE/tmp_data",
                   help="Root output directory for the dataset")
    p.add_argument("--num_episodes", type=int, default=500,
                   help="Number of episodes to process")
    p.add_argument("--langgeonet_ckpt",
                   default="costmap_predictor/langgeonet/checkpoints/best_model.pt",
                   help="Path to LangGeoNet checkpoint (.pt)")
    p.add_argument("--config",
                   default="vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml",
                   help="Habitat config YAML")
    p.add_argument("--image_size", type=int, nargs=2, default=[120, 160],
                   metavar=("H", "W"),
                   help="Output image resolution (height width)")
    p.add_argument("--cost_scale", type=float, default=100.0,
                   help="Scale factor for [0,1] distances → path-length costs")
    p.add_argument("--use_gt_geodesic", action="store_true",
                   help="Use GT geodesic distances instead of LangGeoNet")
    p.add_argument("--train_ratio", type=float, default=0.9,
                   help="Fraction of trajectories for the train split")
    p.add_argument("--h5_name", default="costmaps.h5",
                   help="Filename of the output HDF5 inside --out_dir")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/test split")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    h5_path = os.path.join(args.out_dir, args.h5_name)

    print(f"Output directory : {args.out_dir}")
    print(f"HDF5 path        : {h5_path}")
    print(f"Image size (H×W) : {args.image_size[0]}×{args.image_size[1]}")
    print(f"Cost scale       : {args.cost_scale}")
    print()

    # ── Habitat environment ─────────────────────────────────────────────────
    config = get_config(config_paths=args.config)
    config.defrost()
    # Minimal measurement set — metrics are not needed for data collection
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.freeze()
    env = VLNCEDaggerEnv(config=config)

    # ── LangGeoNet predictor (only when not using GT geodesic) ─────────────
    predictor = None
    if not args.use_gt_geodesic:
        predictor = LangGeoNetPredictor(args.langgeonet_ckpt)

    # ── Dataset builder ─────────────────────────────────────────────────────
    builder = DatasetBuilder(out_dir=args.out_dir, h5_path=h5_path)

    try:
        traj_names = collect_and_save(
            env=env,
            builder=builder,
            num_episodes=args.num_episodes,
            image_size=tuple(args.image_size),
            cost_scale=args.cost_scale,
            predictor=predictor,
            use_gt_geodesic=args.use_gt_geodesic,
        )
    finally:
        env.close()

    if traj_names:
        print("\nWriting split files …")
        write_split_files(args.out_dir, traj_names,
                          train_ratio=args.train_ratio,
                          seed=args.seed)

    print(f"\nDone.  {len(traj_names)} trajectories saved to '{args.out_dir}'")
    print(f"Costmap HDF5 : '{h5_path}'")


if __name__ == "__main__":
    main()
