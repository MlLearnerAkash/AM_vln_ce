import os
from collections import defaultdict
import torch
import tqdm
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat import logger

from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
from habitat_extensions.utils import generate_video, observations_to_image
from habitat.utils.visualizations.utils import append_text_to_image
from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.config.default import get_config
from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances, encode_normalized_distances_to_frame, save_norm_frame_heatmap


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

            frames = []
            norm_frames = []

            for point in reference_path:
                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    obs, _, done, info = env.step(best_action)
                    frame = observations_to_image(obs, info)
                    #NOTE: Need to change the frame
                    frame= obs["rgb"]
                    semantic_frame = obs["semantic"]
                    #NOTE: need to change the inf value to some other value
                    distances = get_object_geodesic_distances(env, semantic_frame)
                    
                    norm_frame = encode_normalized_distances_to_frame(
                        semantic_frame, distances
                    )

                    frames.append(frame)
                    norm_frames.append(norm_frame)

                    if done:
                        break

                if env._env.episode_over:
                    break

            yield {
                "episode_id": episode_id,
                "instruction": instruction,
                "frames": frames,
                "norm_frames": norm_frames,
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