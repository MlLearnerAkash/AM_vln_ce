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

sys.path.append("/data/ws")
from obj_rec_to_rank_predictor.src.rank_predictor import RankPredictor
from obj_rec_to_rank_predictor.src.object_react import ObjRelLearntController

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

def save_numpy_images_to_temp(frames, norm_frames, temp_dir):
    frames_dir = os.path.join(temp_dir, "frames")
    norm_frames_dir = os.path.join(temp_dir, "norm_frames")
    heatmap_dir= os.path.join(temp_dir, "heatmap")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(norm_frames_dir, exist_ok=True)

    frame_paths = []
    norm_frame_paths = []

    for idx, (frame, norm_frame) in enumerate(zip(frames, norm_frames)):
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

        save_norm_frame_heatmap(frame, norm_frame, heatmap_dir, f"norm_frame_{idx}.png")
    return frame_paths, norm_frame_paths

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

def get_goal_image(image, heatmap, patch_grid=(7, 7), target_size=(21, 16), dims=8):
    H, W = image.shape[:2]
    patch_h = H // patch_grid[0]
    patch_w = W // patch_grid[1]
    device = heatmap.device
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Calculate number of patches
    num_patches = patch_grid[0] * patch_grid[1]  # 49
    
    # Create masks array (49, 120, 160) - target size scaled
    target_h, target_w = 120, 160  # H/4, W/4
    masks = np.zeros((num_patches, target_h, target_w), dtype=np.uint8)
    
    # Calculate patch size in target coordinates
    target_patch_h = target_h // patch_grid[0]
    target_patch_w = target_w // patch_grid[1]
    
    # Create mask for each patch
    patch_idx = 0
    for i in range(patch_grid[0]):
        for j in range(patch_grid[1]):
            # Calculate patch boundaries in target space
            y_start = i * target_patch_h
            y_end = (i + 1) * target_patch_h if i < patch_grid[0] - 1 else target_h
            x_start = j * target_patch_w
            x_end = (j + 1) * target_patch_w if j < patch_grid[1] - 1 else target_w
            
            # Set mask to 1 only for this patch region
            masks[patch_idx, y_start:y_end, x_start:x_end] = 1
            
            patch_idx += 1
    
    # Calculate PLS from heatmap (costmap logic)
    cost_values = heatmap[0].flatten().cpu().numpy()  # (49,)
    pls = (cost_values * 200).astype(np.uint8)  # Scale to [0, 200]
    return masks, pls



if __name__ == "__main__":
    NUM_EPOCHS= 1000
    ACCUMULATION_STEPS = 1
    PREPROCESS_GT = False  # Set to True to apply bilateral filtering to GT heatmaps
    
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
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["TOP_DOWN_MAP_VLNCE"]
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



    for epoch in range(NUM_EPOCHS):
        for episode_data in episode_generator(env, num_episodes=2):
            episode_id = episode_data["episode_id"]
            instruction = episode_data["instruction"]
            frames = episode_data["frames"]
            norm_frames = episode_data["norm_frames"]

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
            
            # Recalculate observations from the restored initial position
            obs = sim.get_sensor_observations()
            
            if len(frames) != len(norm_frames):
                print(f"Skipping episode {episode_id}: image and heatmap sequence lengths do not match.")
                continue
            step_count= 0
            epoch_loss= []
            video_frames = []  # Collect frames for video generation
            
            with tempfile.TemporaryDirectory(prefix=f"episode_{episode_id}_") as temp_dir:
                frame_paths, norm_frame_paths = save_numpy_images_to_temp(frames, norm_frames, temp_dir)
                instruction = episode_data["instruction"]


                for frame_path, heatmap_path in zip(frame_paths, norm_frame_paths):
                    frame = np.array(Image.open(frame_path))

                    # Forward pass for a single image
                    pred_heatmap = rank_predictor.generate_heatmap(frame, instruction)
                    mask, pls= get_goal_image(frame, pred_heatmap)

                    v, w, vis_img= object_react_controller.predict(frame, (mask, pls))
                    agent, sim, collided = apply_velocity(vel_control, agent, sim, velocity=v, steer=-w, time_step=0.1)
                    
                    #Forces the environment to recalculate all metrics based on current simulator state.
                    env._env._task.measurements.update_measures(
                        episode=env._env.current_episode, 
                        action={"action": "VELOCITY_CONTROL"},  # Dummy action
                        task=env._env.task
                    )
                    env._env._update_step_stats()
                    
                    # Get observations and metrics after velocity update
                    obs = sim.get_sensor_observations()
                    info = env._env.get_metrics()
                    
                    #Adding Vis_img
                    obs['vis_img'] = vis_img
                    # Create video frame
                    video_frame = observations_to_image(obs, info)
                    video_frame = append_text_to_image(video_frame, instruction)
                    video_frames.append(video_frame)
                    
                    step_count += 1
                    
                    # Log collision
                    if collided:
                        logger.info(f"Collision detected at step {step_count}")
            
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

    wandb.finish()