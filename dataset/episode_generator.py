
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
from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances, encode_normalized_distances_to_frame, save_norm_frame_heatmap

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
            for point in reference_path:
                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    obs, _, done, info = env.step(best_action)
                    
                    
                    frame= obs["rgb"]
                    semantic_frame = obs["semantic"]
                    # semantic_frame= make_semantic(obs['semantic'])
                    #NOTE: To visualize the semantics
                    # alpha = 0.7
                    # rgb = np.asarray(frame).astype(np.float32)
                    # sem = np.asarray(semantic_frame).astype(np.int32)
                    
                    # h, w = sem.shape[:2]
                    # colored_sem = np.zeros((h, w, 3), dtype=np.float32)
                    
                    # # Color by CATEGORY (not instance)
                    # unique_ids = np.unique(sem)
                    # for obj_id in unique_ids:
                    #     if obj_id == 0:
                    #         # Unlabeled pixels
                    #         colored_sem[sem == obj_id] = category_to_color['unlabeled']
                    #     elif obj_id in id_to_category:
                    #         # Get category and its color
                    #         category = id_to_category[obj_id]
                    #         color = category_to_color.get(category, category_to_color['unlabeled'])
                    #         colored_sem[sem == obj_id] = color
                    #     else:
                    #         # Unknown ID
                    #         colored_sem[sem == obj_id] = category_to_color['unlabeled']
                    
                    # # Blend and convert to uint8
                    # overlay = (1 - alpha) * rgb + alpha * colored_sem
                    # overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                    # # Draw small category names at instance centroids
                    # overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 0.45
                    # thickness = 1

                    # for obj_id in unique_ids:
                    #     if obj_id == 0:
                    #         continue
                    #     ys, xs = np.where(sem == obj_id)
                    #     if ys.size == 0:
                    #         continue
                    #     cy = int(np.mean(ys))
                    #     cx = int(np.mean(xs))

                    #     category = id_to_category.get(int(obj_id), "unlabeled")
                    #     text = category.replace("_", " ")
                    #     if len(text) > 20:
                    #         text = text[:17] + "..."

                    #     (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    #     tl_x = max(0, cx - text_w // 2 - 2)
                    #     tl_y = max(0, cy - text_h // 2 - 2)
                    #     br_x = min(overlay_bgr.shape[1] - 1, tl_x + text_w + 4)
                    #     br_y = min(overlay_bgr.shape[0] - 1, tl_y + text_h + 4)

                    #     cv2.rectangle(overlay_bgr, (tl_x, tl_y), (br_x, br_y), (0, 0, 0), cv2.FILLED)
                    #     cv2.putText(
                    #         overlay_bgr,
                    #         text,
                    #         (tl_x + 2, br_y - 3),
                    #         font,
                    #         font_scale,
                    #         (255, 255, 255),
                    #         thickness,
                    #         cv2.LINE_AA,
                    #     )

                    # overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


                    # out_dir = "/data/ws/VLN-CE/dataset/semantics"
                    # overlay_path = f"{out_dir}/semantic_overlay_ep{episode_id}_step{len(frames)}.png"
                    # plt.imsave(overlay_path, semantic_frame)
                    # #NOTE: need to change the inf value to some other value
                    # frame = observations_to_image(obs, info)
                    distances = get_object_geodesic_distances(env, semantic_frame)
                    
                    norm_frame = encode_normalized_distances_to_frame(
                        semantic_frame, distances
                    )

                    frames.append(frame)
                    norm_frames.append(norm_frame)
                    semantic_frames.append(semantic_frame)

                    if done:
                        break

                if env._env.episode_over:
                    break

            yield {
                "episode_id": episode_id,
                "instruction": instruction,
                "frames": frames,
                "norm_frames": norm_frames,
                "semantic_frames": semantic_frames
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