import sys
import os
import time

from models.model import QwenVLHeatmapModel, count_parameters
from dataset.episode_generator import episode_generator


from vlnce_baselines.config.default import get_config

if __name__ == "__main__":
    config_path = "/data/ws/VLN-CE/models/configs/train.yaml"
    model = QwenVLHeatmapModel(config_path)
    total_params, trainable_params = count_parameters(model.vl_model)
    print(f"Total parameters in decoder: {total_params}")
    print(f"Trainable parameters in decoder: {trainable_params}")


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

        # instruction = "Go to the kitchen and find a cup on the table."
        image_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
        heatmap_sequence = [f"/data/ws/VLN-CE/reference_path_videos/test_heatmap_{i}.png" for i in range(4)]
        current_timestep = 2
        history_length = 1

        heatmap = model.forward(
            instruction=instruction,
            image_sequence=image_sequence,
            heatmap_sequence=heatmap_sequence,
            current_idx=current_timestep,
            history_length=history_length
        )
        print(f"Heatmap shape: {heatmap.shape}")  # (1, 1, 640, 480)

        
