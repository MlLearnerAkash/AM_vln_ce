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

def reference_path_example_with_video():
    r"""Generate videos of an agent following the VLN-CE reference path."""
    config = get_config(config_paths="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS = ["TOP_DOWN_MAP_VLNCE"]
    config.VIDEO_DIR = "/data/ws/VLN-CE/reference_path_videos"
    config.VIDEO_OPTION = ["disk"]
    config.freeze()

    # Create video directory
    os.makedirs(config.VIDEO_DIR, exist_ok=True)

    env = VLNCEDaggerEnv(config=config)
    follower = ShortestPathFollowerCompat(
        env._env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = "geodesic_path"

    pf = env._env.sim.pathfinder

    actions = defaultdict(list)
    rgb_frames = []
    stats_episodes = {}

    num_episodes = 3
    with tqdm.tqdm(total=num_episodes, desc="Generating reference path videos") as pbar:
        for episode in range(num_episodes):
            obs = env.reset()
            episode_id = env.current_episode.episode_id
            reference_path = env.current_episode.reference_path
            instruction_text = env.current_episode.instruction.instruction_text            
            
            rgb_frames = []
            step_count = 0

            logger.info(f"\n{'='*80}")
            logger.info(f"Episode: {episode_id}")
            logger.info(f"Instruction: {instruction_text}")
            logger.info(f"{'='*80}\n")

            for waypoint_idx, point in enumerate(reference_path):
                
                logger.info(f"Moving to waypoint {waypoint_idx + 1}/{len(reference_path)}")
                while not env._env.episode_over:
                    # current_distance = env._env.sim.geodesic_distance(
                    #         env._env.sim.get_agent_state().position, point
                    #     )


                    # logger.info(f"Waypoint {waypoint_idx + 1}: Distance = {current_distance:.4f}m")
                    #NOTE: braking if collison detected
                    # if env._env.sim.previous_step_collided:
                    #     print(f"Collision detected at: {waypoint_idx + 1} -- agent can't move anymore")
                    #     break
                    best_action = follower.get_next_action(point)
                    if best_action == None:
                        logger.info(f"waypoint {waypoint_idx+1} reached")
                        break

                    obs, _, done, info = env.step(best_action)
                    actions[episode_id].append(best_action)
                    step_count += 1

                    # Capture frame for video
                    frame = observations_to_image(obs, info)
                    frame = append_text_to_image(frame, instruction_text)
                    rgb_frames.append(frame)

                    #NOTE: To collect geodesic distances from current frame objects to goal position
                    semantic_frame = obs["semantic"]
                    distances= get_object_geodesic_distances(env, semantic_frame)
                    norm_frame= encode_normalized_distances_to_frame(semantic_frame, distances)
                    #NOTE: For debug purpose only
                    save_norm_frame_heatmap(norm_frame, "/data/ws/VLN-CE/reference_path_videos/test_heatmap.png")
                    if done:
                        break
                if env._env.episode_over:
                    break
                        
            # Add final STOP action
            # obs, _, done, info = env.step(HabitatSimActions.STOP)
            actions[episode_id].append(HabitatSimActions.STOP)
            frame = observations_to_image(obs, info)
            frame = append_text_to_image(frame, instruction_text)
            rgb_frames.append(frame)
            # Store episode stats
            stats_episodes[episode_id] = {
                "steps": step_count,
                "success": info.get("success", False),
                "spl": info.get("spl", 0.0),
            }

            logger.info(f"Episode Complete: {episode_id}")
            logger.info(f"Steps taken: {step_count}")
            logger.info(f"Success: {info.get('success', False)}")
            logger.info(f"SPL: {info.get('spl', 0.0):.4f}")

            # Generate video
            if len(rgb_frames) > 0:
                generate_video(
                    video_option=config.VIDEO_OPTION,
                    video_dir=config.VIDEO_DIR,
                    images=rgb_frames,
                    episode_id=episode_id,
                    checkpoint_idx=0,
                    metrics={"spl": stats_episodes[episode_id]["spl"]},
                    tb_writer=None,
                )
                logger.info(f"Video saved for episode: {episode_id}")

            pbar.update()

    logger.info("\nActions taken in episodes:")
    for k, v in actions.items():
        logger.info(f"{k}: {len(v)} actions")

    logger.info(f"\nVideos saved to: {config.VIDEO_DIR}")


if __name__ == "__main__":
    reference_path_example_with_video()