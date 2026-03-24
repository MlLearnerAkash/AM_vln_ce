
import os
from collections import defaultdict
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat import logger
import cv2
from habitat_sim.utils.common import d3_40_colors_rgb

from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image
from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.config.default import get_config
from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances, encode_normalized_distances_to_frame, save_norm_frame_heatmap, encode_directional_cue_to_frame

def make_semantic(semantic_obs):
    """Convert semantic IDs to RGB colors while preserving spatial layout"""
    # Use fancy indexing - this preserves the (H, W) spatial structure
    semantic_obs = np.asarray(semantic_obs, dtype=np.int32)
    colors = np.asarray(d3_40_colors_rgb, dtype=np.uint8)
    semantic_image = colors[semantic_obs % 40]  # Output is (H, W, 3)
    return semantic_image

def episode_generator(env, num_episodes=10):
    """
    Generator that yields episode data one at a time.
    Yields per-episode frames and norm_frames without loading all episodes into memory.
    """
    follower = ShortestPathFollowerCompat(
        env._env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = "geodesic_path"
    
    
    with tqdm.tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        for episode_idx in range(num_episodes):
            obs = env.reset()
            episode_id = env.current_episode.episode_id
            reference_path = env.current_episode.reference_path
            instruction = env.current_episode.instruction.instruction_text
            
            semantic_scene = env._env.sim.semantic_annotations()
            id_to_category = {}
            
            for obj in semantic_scene.objects:
                try:
                    obj_id = int(obj.id.split('_')[-1]) if '_' in obj.id else int(obj.id)
                    id_to_category[obj_id] = obj.category.name()
                except:
                    pass
            
            # Get all unique categories in the scene and assign colors
            all_categories = sorted(set(id_to_category.values()))
            category_to_color = {}
            
            if len(all_categories) > 0:
                cmap = plt.get_cmap("tab20" if len(all_categories) <= 20 else "hsv")
                colors = (cmap(np.linspace(0, 1, len(all_categories)))[:, :3] * 255).astype(np.float32)
                
                for idx, category in enumerate(all_categories):
                    category_to_color[category] = colors[idx]
            
            # Gray for unlabeled
            category_to_color['unlabeled'] = np.array([128, 128, 128], dtype=np.float32)

            frames = []
            norm_frames = []
            semantic_frames= []
            gt_actions = []
            direction_cue_frames = []
            direction_weight_maps = []
            prev_semantic_frame = None
            for point in reference_path:
                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    obs, _, done, info = env.step(best_action)
                    
                    
                    frame= obs["rgb"]
                    semantic_frame = obs["semantic"]
                    
                    distances = get_object_geodesic_distances(env, semantic_frame)
                    
                    norm_frame = encode_normalized_distances_to_frame(
                        semantic_frame, distances
                    )

                    # Apply next action
                    next_action = follower.get_next_action(point)
                    weight_map, cue_frame = encode_directional_cue_to_frame(
                        semantic_frame, next_action if next_action is not None else best_action
                    )

                    frames.append(frame)
                    norm_frames.append(norm_frame)
                    semantic_frames.append(semantic_frame)
                    gt_actions.append(best_action)
                    direction_cue_frames.append(cue_frame)
                    direction_weight_maps.append(weight_map)

                    if done:
                        break

                if env._env.episode_over:
                    break

            yield {
                "episode_id": episode_id,
                "instruction": instruction,
                "frames": frames,
                "norm_frames": norm_frames,
                "semantic_frames": semantic_frames,
                "gt_actions": gt_actions,
                "direction_cue_frames": direction_cue_frames,   # [N, H, W, 3] uint8
                "direction_weight_maps": direction_weight_maps, # [N, H, W] float32
            }

            pbar.update()


if __name__ == "__main__":
    config = get_config(
        config_paths="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml"
    )
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["TOP_DOWN_MAP_VLNCE"]
    config.VIDEO_DIR = "/data/ws/VLN-CE/reference_path_videos"
    config.VIDEO_OPTION = ["disk"]
    config.freeze()

    os.makedirs(config.VIDEO_DIR, exist_ok=True)

    for episode_data in episode_generator(config, num_episodes=5):
        episode_id = episode_data["episode_id"]
        instruction = episode_data["instruction"]
        frames = episode_data["frames"]
        norm_frames = episode_data["norm_frames"]

        logger.info(f"\nEpisode: {episode_id}")
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Frames collected: {len(frames)}")
        logger.info(f"Norm frames collected: {len(norm_frames)}")