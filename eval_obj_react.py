import sys
import os
import time
import logging
import wandb



import tempfile
import os
from PIL import Image
import numpy as np
import sys
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

import habitat_sim

from models.model import QwenVLHeatmapModel, count_parameters
from models.unet_clip import CLIPUNet2D
from dataset.episode_generator import episode_generator
from vlnce_baselines.config.default import get_config
from utils.obj_react_velo_to_action import apply_velocity

from vlnce_baselines.common.environments import VLNCEDaggerEnv
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image
from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances, encode_normalized_distances_to_frame


# #Discrete object react
# from controller.object_react.train.inference import load_config, build_model, load_checkpoint
# from controller.object_react.train.inference import predict_from_image_masks_costs
# object_rec_dis_config = load_config("/data/ws/VLN-CE/controller/object_react/train/config/object_react_vln.yaml")
# object_rec_dis_model  = build_model(object_rec_dis_config)
# object_rec_dis_model  = load_checkpoint(object_rec_dis_model, "/data/ws/VLN-CE/controller/object_react/train/logs/object_react_vln/object_react_vln_2026_03_09_08_12_04_/latest.pth", "gnm")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# object_rec_dis_model  = object_rec_dis_model.to(device).eval()

sys.path.append("/data/ws")
from obj_rec_to_rank_predictor.src.rank_predictor import RankPredictor
from obj_rec_to_rank_predictor.src.object_react import ObjRelLearntController

from costmap_predictor.langgeonet.inference import LangGeoNetPredictor

def save_norm_frame_heatmap(image, heatmap, save_dir, filename, alpha=0.5):
    """
    Overlay a heatmap (values in [0,255]) on top of an RGB image and save to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Normalize heatmap to [0, 255] and convert to uint8
    hm = heatmap.astype(np.float32)
    hm = (hm - hm.min()) / (hm.ptp() + 1e-8)
    hm_uint8 = np.uint8(hm * 255)
    # Apply colormap
    heatmap_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_VIRIDIS)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # Ensure image is uint8 in [0,255]
    if image.dtype != np.uint8:
        img = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    else:
        img = image
    # Resize heatmap if needed
    if heatmap_color.shape[:2] != img.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    # Overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return save_path

def save_numpy_images_to_temp(frames,semantic_frames, norm_frames, temp_dir):
    frames_dir = os.path.join(temp_dir, "frames")
    norm_frames_dir = os.path.join(temp_dir, "norm_frames")
    semantic_dir= os.path.join(temp_dir, "semantics")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(norm_frames_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok= True)

    frame_paths = []
    norm_frame_paths = []
    semantic_paths= []

    for idx, (frame, semantic_frame, norm_frame) in enumerate(zip(frames, semantic_frames, norm_frames)):
        frame_path = os.path.join(frames_dir, f"frame_{idx}.png")
        if isinstance(frame, np.ndarray):
            Image.fromarray(frame).save(frame_path)
        frame_paths.append(frame_path)

        norm_path = os.path.join(norm_frames_dir, f"norm_frame_{idx}.png")
        if isinstance(norm_frame, np.ndarray):
            arr = norm_frame
            if arr.dtype != np.uint8:
                arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-8)).astype(np.uint8)
            Image.fromarray(arr).save(norm_path)
        norm_frame_paths.append(norm_path)

        semantic_path = os.path.join(semantic_dir, f"semantic_{idx}.npy")
        if isinstance(semantic_frame, np.ndarray):
            np.save(semantic_path, semantic_frame.astype(np.int32))  # preserve exact IDs
        semantic_paths.append(semantic_path)

        # save_norm_frame_heatmap(frame, norm_frame, heatmap_dir, f"norm_frame_{idx}.png")
    return frame_paths, semantic_paths, norm_frame_paths

def prepare_ground_truth_heatmap(heatmap_tensor):
    """
    Optional preprocessing for ground truth heatmaps.
    Applies bilateral filtering to ensure smooth intra-region values
    while preserving boundaries.
    
    Args:
        heatmap_tensor: [B, 1, H, W] tensor in range [0, 1]
    
    Returns:
        Filtered heatmap tensor
    """
    import cv2
    
    filtered_batch = []
    for i in range(heatmap_tensor.shape[0]):
        heatmap = heatmap_tensor[i, 0].cpu().numpy()  # [H, W]
        
        # Convert to uint8 for cv2
        heatmap_8bit = (heatmap * 255).astype(np.uint8)
        
        # Apply bilateral filter: smooths while preserving edges
        filtered = cv2.bilateralFilter(heatmap_8bit, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Convert back to float32 [0, 1]
        filtered_float = filtered.astype(np.float32) / 255.0
        filtered_batch.append(torch.from_numpy(filtered_float))
    
    # Stack back to batch
    filtered_tensor = torch.stack(filtered_batch).unsqueeze(1)  # [B, 1, H, W]
    return filtered_tensor.to(heatmap_tensor.device)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_goal_image(image, heatmap, patch_grid=(7, 7), target_size=(120, 160), dims=8):
    target_h, target_w = target_size

    # ── Step 1: Scale heatmap patch values to [0, 200] ──────────────────────
    cost_values = heatmap[0].flatten().cpu().numpy()        # (49,)
    pls = (cost_values * 200).astype(np.float32)            # (49,) in [0, 200]

    # ── Step 2: Resize semantic map → target_size (NEAREST keeps integer IDs) 
    semantic_resized = cv2.resize(
        semantic_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)                                      # (target_h, target_w)

    # ── Step 3: Build spatial patch costmap (target_h, target_w) ────────────
    patch_costmap = np.zeros((target_h, target_w), dtype=np.float32)
    target_patch_h = target_h // patch_grid[0]
    target_patch_w = target_w // patch_grid[1]

    patch_idx = 0
    for i in range(patch_grid[0]):
        for j in range(patch_grid[1]):
            y_start = i * target_patch_h
            y_end   = (i + 1) * target_patch_h if i < patch_grid[0] - 1 else target_h
            x_start = j * target_patch_w
            x_end   = (j + 1) * target_patch_w if j < patch_grid[1] - 1 else target_w
            patch_costmap[y_start:y_end, x_start:x_end] = pls[patch_idx]
            patch_idx += 1

    # ── Step 4: Get unique objects — skip background (ID = 0) ───────────────
    unique_objects = [int(obj) for obj in np.unique(semantic_resized) if obj != 0]
    N = len(unique_objects)                                 # number of objects in frame
    object_cost_pairs = []                                  # [(obj_id, cost), ...]

    for obj_id in unique_objects:
        obj_mask = (semantic_resized == obj_id)             # (target_h, target_w) bool
        aggregated_cost = float(np.mean(patch_costmap[obj_mask]))
        object_cost_pairs.append((obj_id, aggregated_cost))

    # ── Step 6: Sort objects by cost DESCENDING ──────────────────────────────
    object_cost_pairs.sort(key=lambda x: x[1], reverse=True)
    # [(obj_id_highest_cost, cost), ..., (obj_id_lowest_cost, cost)]

    # ── Step 7: Build (N, target_h, target_w) costmap in descending order ───
    costmap = np.zeros((N, target_h, target_w), dtype=np.float32)
    sorted_object_ids  = []
    sorted_object_costs = []

    for ch_idx, (obj_id, aggregated_cost) in enumerate(object_cost_pairs):
        obj_mask = (semantic_resized == obj_id)
        costmap[ch_idx][obj_mask] = aggregated_cost        # fill object pixels with cost
        sorted_object_ids.append(obj_id)
        sorted_object_costs.append(aggregated_cost)

    # pls: patch costs in ascending order  → (49,)  [low → high]
    # sorted_object_costs: per-object costs in descending order → (N,) [high → low]
    pls_assigned = np.array(sorted_object_costs, dtype=np.float32)  # only assigned values

    return costmap, pls_assigned


def get_goal_image_langgeonet(
    image,
    semantic_map,
    instruction,
    predictor,
    target_size=(120, 160),
):
    """
    Drop-in replacement for get_goal_image that obtains per-object costs via
    LangGeoNetPredictor instead of a patch heatmap.

    Steps
    -----
    1. Resize ``semantic_map`` to ``target_size`` (nearest-neighbour, preserves IDs).
    2. Build one binary mask per unique object (background ID 0 excluded).
    3. Run ``predictor.predict_frame`` → per-object normalised geodesic distances [K].
    4. Sort objects by distance *descending* (highest-cost / farthest first),
       matching the convention of ``get_goal_image``.
    5. Return the same (costmap, pls_assigned) shapes:
         costmap      : np.ndarray  [N, target_h, target_w]  per-object cost channels
         pls_assigned : np.ndarray  [N,]                     per-object scalar costs

    Args
    ----
    image        : np.ndarray [H, W, 3]  RGB frame (uint8 or float)
    semantic_map : np.ndarray [H, W]     integer instance-segmentation map
    instruction  : str                   navigation instruction
    predictor    : LangGeoNetPredictor   pre-loaded predictor instance
    target_size  : (int, int)            (height, width) of output costmap

    Returns
    -------
    costmap      : np.ndarray [N, target_h, target_w]
    pls_assigned : np.ndarray [N,]
    """
    target_h, target_w = target_size

    # ── Step 1: Resize semantic map ──────────────────────────────────────────
    semantic_resized = cv2.resize(
        semantic_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)                                      # (target_h, target_w)

    # ── Step 2: Collect unique objects (skip background = 0) ────────────────
    unique_objects = [int(obj) for obj in np.unique(semantic_resized) if obj != 0]
    N = len(unique_objects)

    if N == 0:
        # No objects detected — return empty arrays
        return np.zeros((0, target_h, target_w), dtype=np.float32), np.array([], dtype=np.float32)

    # Build binary masks  [N, target_h, target_w]
    masks = np.stack(
        [(semantic_resized == obj_id).astype(np.float32) for obj_id in unique_objects],
        axis=0,
    )                                                       # (N, target_h, target_w)

    # ── Step 3: Run LangGeoNet ───────────────────────────────────────────────
    # predict_frame expects image as PIL or uint8 ndarray
    distances, _flat_costmap, _attn = predictor.predict_frame(
        image, masks, instruction
    )                                                       # distances: (N,) in [0, 1]

    # Scale to [0, 200] to match the cost range used in get_goal_image
    costs = (distances * 200.0).astype(np.float32)         # (N,) in [0, 200]

    # ── Step 4: Pair objects with costs and sort descending ──────────────────
    object_cost_pairs = list(zip(range(N), costs.tolist()))
    object_cost_pairs.sort(key=lambda x: x[1], reverse=True)
    # [(local_idx, cost), ...] highest cost first

    # ── Step 5: Build (N, target_h, target_w) costmap ───────────────────────
    costmap = np.zeros((N, target_h, target_w), dtype=np.float32)
    sorted_object_costs = []

    for ch_idx, (local_idx, aggregated_cost) in enumerate(object_cost_pairs):
        obj_mask = masks[local_idx] > 0                     # (target_h, target_w) bool
        costmap[ch_idx][obj_mask] = aggregated_cost
        sorted_object_costs.append(aggregated_cost)

    pls_assigned = np.array(sorted_object_costs, dtype=np.float32)  # (N,) descending

    return costmap, pls_assigned

#from waypoints to discrete actions
def decide_action(waypoints,
                  turn_thresh_deg=15.0,
                  stop_thresh=0.1):
    
    waypoints = np.array(waypoints)
    waypoints= waypoints[:10] #taking first 3 future points
    dists = waypoints[:, 0]  

    if dists[0] < stop_thresh:
        return "STOP", 0.0

    col1 = waypoints[:, 1]
    slope = np.polyfit(waypoints[:, 0], col1, 1)[0]

    mid_idx = len(waypoints) // 4
    lateral = col1[mid_idx]
    angle_deg = np.degrees(np.arctan2(abs(lateral), waypoints[mid_idx, 0]))

    if abs(slope) < np.tan(np.radians(turn_thresh_deg)):
        action = "MOVE_FORWARD"
    elif slope > 0:
        action = "TURN_LEFT"   
    else:
        action = "TURN_RIGHT"  

    return action, slope

#move agents by waypoints
import habitat_sim
from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

def move_agent_by_waypoint(wp, agent, sim, time_step=0.1, num_steps=1):
    """
    Move agent through the first `num_steps` waypoints in the local (dx, dy) frame.
    wp: np.ndarray [N, 2], wp[i] = (cumulative_forward, cumulative_lateral)
    Since coordinates are cumulative, the incremental delta at step i is wp[i] - wp[i-1].
    """
    n = min(num_steps, len(wp))
    prev = np.array([0.0, 0.0])  # cumulative origin

    for i in range(n):
        curr = wp[i]
        delta = curr - prev          # incremental displacement from last position
        dx = float(delta[0])         # forward
        dy = float(delta[1])         # lateral (+left in GNM)
        prev = curr

        state = agent.state

        forward_vec = habitat_sim.utils.quat_rotate_vector(
            state.rotation, np.array([0, 0, -1.0])
        )
        left_vec = habitat_sim.utils.quat_rotate_vector(
            state.rotation, np.array([-1.0, 0, 0])
        )

        target_pos = state.position + forward_vec * dx + left_vec * dy

        yaw = np.arctan2(dy, dx)
        delta_rot = habitat_sim.utils.quat_from_angle_axis(yaw, np.array([0, 1.0, 0]))
        new_rotation = delta_rot * state.rotation

        end_pos = sim.step_filter(state.position, target_pos)

        state.position = end_pos
        state.rotation = quat_from_magnum(quat_to_magnum(new_rotation))
        agent.set_state(state)

        sim.step_physics(dt=time_step)

    return agent, sim

if __name__ == "__main__":
    NUM_EPOCHS= 1
    ACCUMULATION_STEPS = 1
    PREPROCESS_GT = False  # Set to True to apply bilateral filtering to GT heatmaps

    # ── Action mode ───────────────────────────────────────────────────────
    # True  → use env.step() with discrete actions (MOVE_FORWARD / TURN_LEFT /
    #          TURN_RIGHT / STOP) derived from decide_action().
    # False → use apply_velocity() with continuous velocity control.
    USE_DISCRETE_ACTIONS = False

    # ── Observation source ────────────────────────────────────────────────
    # True  → use live RGB + semantic from the simulator at each step and
    #          compute geodesic heatmaps on-the-fly.
    # False → use pre-collected GT frames / norm_frames from episode_generator.
    USE_LIVE_OBS = True

    # Discrete-action map: decide_action() labels → Habitat action names
    DISCRETE_ACTION_MAP = {
        "MOVE_FORWARD": "MOVE_FORWARD",
        "TURN_LEFT":    "TURN_LEFT",
        "TURN_RIGHT":   "TURN_RIGHT",
        "STOP":         "STOP",
    }

    # ── Velocity gains ────────────────────────────────────────────────────
    # Calibrate with: python calibrate_vel_gains.py --mode analytic
    # Root cause of overshoot: displacement = v * TRANSLATION_GAIN
    # Optimal value:  TRANSLATION_GAIN = ref_path_step_size / mean(v_gnm)
    TRANSLATION_GAIN = 10.0   # ← replace with calibrate_vel_gains.py output
    ROTATION_GAIN    = 2.0    # ← replace with calibrate_vel_gains.py output
    
    config_path = "/data/ws/VLN-CE/models/configs/train.yaml"
    # model = QwenVLHeatmapModel(config_path)
    model = CLIPUNet2D(in_channels=3, out_channels=1, fChannel=64).to("cuda" if torch.cuda.is_available() else "cpu")
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters in decoder: {total_params}")
    print(f"Trainable parameters in decoder: {trainable_params}")



    rank_predictor_model_config= load_config('/data/ws/obj_rec_to_rank_predictor/configs/rank_predictor.yaml')
    rank_predictor = RankPredictor(rank_predictor_model_config['model'], device=rank_predictor_model_config['model']['device'])

    object_react_config= "/data/ws/obj_rec_to_rank_predictor/configs/object_react.yaml"
    object_react_controller= ObjRelLearntController(object_react_config, dirname_vis_episode= "/data/dataset/RXR/dataset/rxr/test")

    langgeonet = LangGeoNetPredictor("/data/ws/VLN-CE/costmap_predictor/langgeonet/checkpoints/best_model.pt")

    logging.basicConfig(
    # filename="train.log",
    # filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    config = get_config(
        config_paths="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml"
    )
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS = [
        "DISTANCE_TO_GOAL",
        "SUCCESS",
        "SPL",
        "SOFT_SPL",
        "PATH_LENGTH",
        "STEPS_TAKEN",
        "NDTW",
        "SDTW",
        "TOP_DOWN_MAP_VLNCE",
    ]
    config.VIDEO_DIR = "/data/ws/VLN-CE/reference_path_videos"
    config.VIDEO_OPTION = ["disk"]
    config.freeze()

    env = VLNCEDaggerEnv(config=config)
    sim= env._env.sim
    agent = env._env.sim.get_agent(0)
    action_names = list(env._env.sim.sim_config.agents[0].action_space.keys())

    # create and configure a new VelocityControl structure
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.controlling_ang_vel = True
    vel_control.ang_vel_is_local = True
    vel_control = vel_control

    os.makedirs(config.VIDEO_DIR, exist_ok=True)

    wandb.init(
        project="gradient_loss_VLNCE-Heatmap",
        name="CLIPUNet-ContinuousHeatmap-Training",
        config={
            "lr": 1e-4,
            "optimizer": "AdamW",
        },
         mode="dryrun"
    )



    episode_metrics = []  # collect per-episode SDTW/NDTW/SPL

    for epoch in range(NUM_EPOCHS):
        for episode_data in episode_generator(env, num_episodes=36):
            episode_id = episode_data["episode_id"]
            instruction = episode_data["instruction"]
            frames = episode_data["frames"]
            norm_frames = episode_data["norm_frames"]
            semantic_frames= episode_data["semantic_frames"]

            # After episode_generator completes, restore agent to initial position
            # Get the episode's start position and rotation
            current_episode = env._env.current_episode
            start_position = current_episode.start_position
            start_rotation = current_episode.start_rotation
            
            # Manually restore agent to start position (no reset needed)
            sim = env._env.sim
            agent = env._env.sim.get_agent(0)
            agent_state = agent.state
            agent_state.position = start_position
            agent_state.rotation = start_rotation
            agent.set_state(agent_state)
            
            # Reset environment's internal state counters
            env._env._elapsed_steps = 0
            env._env._episode_over = False
            
            # Clear trajectory history and reset all measurements (this clears the blue line)
            env._env._task.measurements.reset_measures(
                episode=current_episode, 
                task=env._env.task
            )
            
            # Initialize agent by executing the first GT action so the agent
            gt_actions = episode_data.get("gt_actions", [])
            if gt_actions:
                obs, _, _, _ = env.step(gt_actions[0])
            else:
                obs = sim.get_sensor_observations()
            
            if len(frames) != len(norm_frames):
                print(f"Skipping episode {episode_id}: image and heatmap sequence lengths do not match.")
                continue
            step_count= 0
            epoch_loss= []
            video_frames = []  # Collect frames for video generation
            
            max_steps = len(frames)  # match GT trajectory length
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if USE_LIVE_OBS:
                # ════════════════════════════════════════════════════════════
                # LIVE OBS MODE: RGB + semantic read from the simulator at
                # each step; heatmaps computed on-the-fly from geodesic dists.
                # ════════════════════════════════════════════════════════════
                for _ in range(max_steps):
                    if env._env.episode_over:
                        break

                    # ── Live RGB + semantic from current simulator state ────
                    frame = np.array(obs["rgb"])
                    semantic_map = obs["semantic"].astype(np.int32)

                    # ── Compute GT heatmap from live geodesic distances ─────
                    distances = get_object_geodesic_distances(env, semantic_map)
                    norm_frame_live = encode_normalized_distances_to_frame(semantic_map, distances)
                    gt_heatmap_img = norm_frame_live.astype(np.float32)
                    gt_heatmap_img = (gt_heatmap_img - gt_heatmap_img.min()) / (gt_heatmap_img.ptp() + 1e-8)
                    gt_heatmap_resized = cv2.resize(gt_heatmap_img, (7, 7), interpolation=cv2.INTER_AREA)
                    gt_heatmap = torch.from_numpy(gt_heatmap_resized).unsqueeze(0).to(device)

                    # ── Forward pass ────────────────────────────────────────
                    pred_heatmap = rank_predictor.generate_heatmap(frame, instruction)
                    # mask, pls = get_goal_image_langgeonet(frame, semantic_map, instruction, langgeonet)
                    mask, pls = get_goal_image(frame, gt_heatmap)  # pred_heatmap

                    wp, v, w, vis_img = object_react_controller.predict(frame, (mask, pls))

                    action, angle_deg = decide_action(wp)

                    if USE_DISCRETE_ACTIONS:
                        # ── Discrete action branch ──────────────────────────
                        habitat_action = DISCRETE_ACTION_MAP.get(action, "MOVE_FORWARD")
                        step_result = env.step({"action": habitat_action})
                        if isinstance(step_result, tuple):
                            obs, _, done, info = step_result
                        else:
                            obs = step_result
                        sim = env._env.sim
                        agent = env._env.sim.get_agent(0)
                        info = env._env.get_metrics()
                        if habitat_action == "STOP" or env._env.episode_over:
                            break
                    else:
                        # ── Velocity control branch ─────────────────────────
                        if np.linalg.norm(wp[0]) < 0.05:
                            break  # reached goal

                        agent, sim = move_agent_by_waypoint(wp, agent, sim)

                        # Update env metrics
                        env._env._task.measurements.update_measures(
                            episode=env._env.current_episode,
                            action={"action": "VELOCITY_CONTROL"},
                            task=env._env.task
                        )
                        env._env._update_step_stats()
                        obs = sim.get_sensor_observations()
                        info = env._env.get_metrics()

                    # ── Build video frame ───────────────────────────────────
                    obs['vis_img'] = vis_img
                    video_frame = observations_to_image(obs, info)
                    video_frame = append_text_to_image(video_frame, instruction)
                    video_frames.append(video_frame)

                    step_count += 1

            else:
                # ════════════════════════════════════════════════════════════
                # GT FRAMES MODE: use pre-collected frames / norm_frames from
                # episode_generator (saved to a temp dir and read back).
                # ════════════════════════════════════════════════════════════
                with tempfile.TemporaryDirectory(prefix=f"episode_{episode_id}_") as temp_dir:
                    frame_paths, semantic_paths, norm_frame_paths = save_numpy_images_to_temp(
                        frames, semantic_frames, norm_frames, temp_dir
                    )

                    for frame_path, semantic_path, heatmap_path in zip(frame_paths, semantic_paths, norm_frame_paths):

                        # NOTE: GT semantic map
                        semantic_map = np.load(semantic_path)
                        frame = np.array(Image.open(frame_path))

                        gt_heatmap_img = np.array(Image.open(heatmap_path).convert("L")).astype(np.float32)
                        gt_heatmap_img = (gt_heatmap_img - gt_heatmap_img.min()) / (gt_heatmap_img.ptp() + 1e-8)
                        gt_heatmap_resized = cv2.resize(gt_heatmap_img, (7, 7), interpolation=cv2.INTER_AREA)
                        gt_heatmap = torch.from_numpy(gt_heatmap_resized).unsqueeze(0).to(device)

                        # Forward pass for a single image
                        pred_heatmap = rank_predictor.generate_heatmap(frame, instruction)
                        # mask, pls = get_goal_image_langgeonet(frame, semantic_map, instruction, langgeonet)
                        mask, pls = get_goal_image(frame, gt_heatmap)  # pred_heatmap

                        wp, v, w, vis_img = object_react_controller.predict(frame, (mask, pls))

                        action, angle_deg = decide_action(wp)

                        if USE_DISCRETE_ACTIONS:
                            # ── Discrete action branch ──────────────────────
                            if env._env.episode_over:
                                break
                            habitat_action = DISCRETE_ACTION_MAP.get(action, "MOVE_FORWARD")
                            step_result = env.step({"action": habitat_action})
                            if isinstance(step_result, tuple):
                                obs, _, done, info = step_result
                            else:
                                obs = step_result
                            sim = env._env.sim
                            agent = env._env.sim.get_agent(0)
                            info = env._env.get_metrics()
                            collided = False
                            if habitat_action == "STOP" or env._env.episode_over:
                                break
                        else:
                            # ── Velocity control branch ─────────────────────
                            if np.linalg.norm(wp[0]) < 0.05:
                                break  # reached goal

                            agent, sim = move_agent_by_waypoint(wp, agent, sim)

                            # Update env metrics
                            env._env._task.measurements.update_measures(
                                episode=env._env.current_episode,
                                action={"action": "VELOCITY_CONTROL"},
                                task=env._env.task
                            )
                            env._env._update_step_stats()
                            obs = sim.get_sensor_observations()
                            info = env._env.get_metrics()

                        # Adding Vis_img
                        obs['vis_img'] = vis_img
                        # Create video frame
                        video_frame = observations_to_image(obs, info)
                        video_frame = append_text_to_image(video_frame, instruction)
                        video_frames.append(video_frame)

                        step_count += 1

                        # Log collision
                        # if collided:
                        #     logger.info(f"Collision detected at step {step_count}")
            
            # Collect per-episode metrics
            final_info = env._env.get_metrics()
            ep_sdtw  = final_info.get("sdtw",  None)
            ep_ndtw  = final_info.get("ndtw",  None)
            ep_spl   = final_info.get("spl",   None)
            ep_softspl = final_info.get("softspl", None)
            ep_succ  = final_info.get("success", None)
            ep_dtg   = final_info.get("distance_to_goal", None)
            episode_metrics.append({
                "episode_id": episode_id,
                "sdtw": ep_sdtw,
                "ndtw": ep_ndtw,
                "spl":  ep_spl,
                "softspl": ep_softspl,
                "success": ep_succ,
                "distance_to_goal": ep_dtg,
                "steps": step_count,
            })
            logger.info(
                f"Episode {episode_id} | SDTW={ep_sdtw:.4f} "
                f"NDTW={ep_ndtw:.4f} SPL={ep_spl:.4f} "
                f"SoftSPL={ep_softspl:.4f} "
                f"Success={ep_succ} DTG={ep_dtg:.2f}m steps={step_count}"
            )

            # Generate video for this episode
            if len(video_frames) > 0:
                generate_video(
                    video_option=config.VIDEO_OPTION,
                    video_dir=config.VIDEO_DIR,
                    images=video_frames,
                    episode_id=f"{episode_id}_epoch{epoch}",
                    checkpoint_idx=0,
                    metrics={"steps": step_count},
                    tb_writer=None,
                )
                logger.info(f"Video saved for episode: {episode_id}, steps: {step_count}")

    # ── Aggregate metrics across all episodes ───────────────────────────────
    if episode_metrics:
        def _has_inf(m):
            return any(
                v is not None and np.isinf(v)
                for k, v in m.items()
                if k != "episode_id" and k != "steps"
            )

        valid_metrics = [m for m in episode_metrics if not _has_inf(m)]
        skipped = len(episode_metrics) - len(valid_metrics)
        if skipped:
            logger.info(f"Skipping {skipped} episode(s) with inf metric values.")

        def _mean(key):
            vals = [m[key] for m in valid_metrics if m[key] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        logger.info("=" * 60)
        logger.info(f"RESULTS over {len(valid_metrics)} episodes (skipped {skipped} with inf):")
        logger.info(f"  SDTW    : {_mean('sdtw'):.4f}")
        logger.info(f"  NDTW    : {_mean('ndtw'):.4f}")
        logger.info(f"  SPL     : {_mean('spl'):.4f}")
        logger.info(f"  SoftSPL : {_mean('softspl'):.4f}")
        logger.info(f"  Success : {_mean('success'):.4f}")
        logger.info(f"  Avg DTG : {_mean('distance_to_goal'):.2f} m")
        logger.info(f"  Avg Steps: {_mean('steps'):.1f}")
        logger.info("=" * 60)

        wandb.log({
            "eval/sdtw": _mean("sdtw"),
            "eval/ndtw": _mean("ndtw"),
            "eval/spl": _mean("spl"),
            "eval/softspl": _mean("softspl"),
            "eval/success": _mean("success"),
            "eval/distance_to_goal": _mean("distance_to_goal"),
        })

    wandb.finish()