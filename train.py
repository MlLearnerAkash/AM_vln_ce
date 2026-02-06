import sys
import os
import time
import logging
import wandb

import tempfile
import os
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


from models.model import QwenVLHeatmapModel, count_parameters
from models.unet_clip import CLIPUNet2D
from dataset.episode_generator import episode_generator
from vlnce_baselines.config.default import get_config


def save_numpy_images_to_temp(frames, norm_frames, temp_dir):
    frames_dir = os.path.join(temp_dir, "frames")
    norm_frames_dir = os.path.join(temp_dir, "norm_frames")
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

    return frame_paths, norm_frame_paths


if __name__ == "__main__":
    NUM_EPOCHS= 1
    config_path = "/data/ws/VLN-CE/models/configs/train.yaml"
    # model = QwenVLHeatmapModel(config_path)
    model = CLIPUNet2D(in_channels=3, out_channels=1, fChannel=64).to("cuda" if torch.cuda.is_available() else "cpu")
    # total_params, trainable_params = count_parameters(model.vl_model)
    # print(f"Total parameters in decoder: {total_params}")
    # print(f"Trainable parameters in decoder: {trainable_params}")

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

    os.makedirs(config.VIDEO_DIR, exist_ok=True)

    wandb.init(
        project="VLNCE-Heatmap",
        name="QwenVLHeatmapModel-Training",
        config={"lr": 1e-3, "optimizer": "Adam"}
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-5)
    for epoch in range(NUM_EPOCHS):
        for episode_data in episode_generator(config, num_episodes=5):
            episode_id = episode_data["episode_id"]
            instruction = episode_data["instruction"]
            frames = episode_data["frames"]
            norm_frames = episode_data["norm_frames"]

            if len(frames) != len(norm_frames):
                print(f"Skipping episode {episode_id}: image and heatmap sequence lengths do not match.")
                continue
            
            with tempfile.TemporaryDirectory(prefix=f"episode_{episode_id}_") as temp_dir:
                frame_paths, norm_frame_paths = save_numpy_images_to_temp(frames, norm_frames, temp_dir)
                for current_timestep in range(len(frame_paths)):
                    frame_np = np.array(Image.open(frame_paths[current_timestep])).astype(np.float32) / 255.0
                    # Convert to tensor and add batch/channel dimensions if needed
                    frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                    frame_tensor = frame_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

                    gt_heatmap = np.array(Image.open(norm_frame_paths[current_timestep])).astype(np.float32) / 255.0
                    #checksum
                    if np.isclose(gt_heatmap.min(), gt_heatmap.max()):
                        logger.warning(f"Skipping timestep {current_timestep} in episode {episode_id}: GT heatmap is constant (min == max == {gt_heatmap.min()})")
                        continue

                    gt_heatmap_tensor = torch.from_numpy(gt_heatmap).unsqueeze(0).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")  # (1, 1, H, W)

                    # pred_heatmap = model.forward(
                    #     instruction=instruction,
                    #     image_sequence=frame_paths,
                    #     heatmap_sequence=norm_frame_paths,
                    #     current_idx=current_timestep,
                    #     history_length=0
                    # )
                    pred_heatmap= model(frame_tensor, [instruction])
                    
                    

                    if pred_heatmap.shape[-2:] != gt_heatmap_tensor.shape[-2:]:
                        gt_heatmap_tensor = F.interpolate(gt_heatmap_tensor, size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=True)

                    mse_loss = F.mse_loss(pred_heatmap.float(), gt_heatmap_tensor.float())
                    ssim_value = ssim(pred_heatmap, gt_heatmap_tensor, data_range=1.0, size_average=True)
                    ssim_loss = 1 - ssim_value  # SSIM is similarity, so 1-SSIM is the loss
                    total_loss= 0.1*mse_loss+ 0.9*ssim_loss
                    optimizer.zero_grad()
                    # mse_loss.backward()
                    # ssim_loss.backward()
                    total_loss.backward()
                    optimizer.step()

                    logger.info(f"Episode {episode_id} | Timestep {current_timestep} | Heatmap shape: {pred_heatmap.shape} | MSE Loss: {total_loss.item():.6f}")
                    wandb.log({
                        "episode_id": episode_id,
                        "timestep": current_timestep,
                        "total_loss": total_loss.item(),
                        "mse_loss": mse_loss.item(),
                        "ssim_loss": ssim_loss.item(),
                        "gt_heatmap": wandb.Image(gt_heatmap * 255),  # scale for visualization
                        "pred_heatmap": wandb.Image(pred_heatmap.detach().cpu().squeeze().numpy() * 255)
                    })
                

    save_path = f"/data/ws/VLN-CE/checkpoints/qwenvl_heatmap_model_ep{episode_id}.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved full model checkpoint to {save_path}")