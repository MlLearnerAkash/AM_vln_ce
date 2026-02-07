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
from torch.utils.data import Dataset, DataLoader


from models.model import QwenVLHeatmapModel, count_parameters
from models.unet_clip import CLIPUNet2D
from dataset.episode_generator import episode_generator
from vlnce_baselines.config.default import get_config
from dataset.episode_dataset import EpisodeDataset, collate_fn
from losses.continuous_heatmap_loss import ContinuousHeatmapLoss, SimplifiedContinuousLoss

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
        name="CLIPUNet-ContinuousHeatmap-Training",
        config={
            "lr": 1e-4,
            "optimizer": "AdamW",
            "loss": "ContinuousHeatmapLoss",
            "mse_weight": 1.0,
            "ssim_weight": 0.8,
            "boundary_weight": 0.5,
            "smooth_weight": 0.3
        }
    )

    # Initialize the improved loss function
    # Option 1: Full loss with boundary and smoothness terms (RECOMMENDED)
    criterion = ContinuousHeatmapLoss(
        mse_weight=1.0,
        ssim_weight=0.8,
        boundary_weight=0.5,
        smooth_weight=0.3,
        boundary_threshold=0.1
    )
    
    # Option 2: Start with simplified version and gradually add complexity
    # criterion = SimplifiedContinuousLoss(mse_weight=1.0, ssim_weight=0.5)
    
    logger.info(f"Using loss function: {criterion.__class__.__name__}")

    # Use AdamW optimizer (better than SGD for this task)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )
    for epoch in range(NUM_EPOCHS):
        for episode_data in episode_generator(config, num_episodes=1):
            episode_id = episode_data["episode_id"]
            instruction = episode_data["instruction"]
            frames = episode_data["frames"]
            norm_frames = episode_data["norm_frames"]

            if len(frames) != len(norm_frames):
                print(f"Skipping episode {episode_id}: image and heatmap sequence lengths do not match.")
                continue
            step_count= 0
            epoch_loss= []
            optimizer.zero_grad()
            with tempfile.TemporaryDirectory(prefix=f"episode_{episode_id}_") as temp_dir:
                frame_paths, norm_frame_paths = save_numpy_images_to_temp(frames, norm_frames, temp_dir)
                frames = [np.array(Image.open(p)) for p in frame_paths]
                heatmaps = [np.array(Image.open(p)) for p in norm_frame_paths]
                instruction = episode_data["instruction"]

                dataset = EpisodeDataset(frames, heatmaps, instruction)
                dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

                for batch_frames, batch_heatmaps, batch_instruction in dataloader:
                    batch_frames = batch_frames.to("cuda" if torch.cuda.is_available() else "cpu")
                    batch_heatmaps = batch_heatmaps.to("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # Optional: Preprocess ground truth for better smoothness
                    if PREPROCESS_GT:
                        batch_heatmaps = prepare_ground_truth_heatmap(batch_heatmaps)
                    
                    # Forward pass
                    pred_heatmap = model(batch_frames, batch_instruction)
                    
                    # Resize target if needed
                    if pred_heatmap.shape[-2:] != batch_heatmaps.shape[-2:]:
                        batch_heatmaps = F.interpolate(
                            batch_heatmaps, 
                            size=pred_heatmap.shape[-2:], 
                            mode='bilinear', 
                            align_corners=True
                        )

                    # Compute loss with new loss function
                    total_loss, loss_dict = criterion(pred_heatmap, batch_heatmaps)

                    epoch_loss.append(total_loss)

                    step_count += 1
                    if step_count % ACCUMULATION_STEPS == 0:
                        # Average accumulated losses
                        loss = torch.stack(epoch_loss).mean()
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        epoch_loss = []

                        # Logging
                        logger.info(
                            f"Episode {episode_id} | Step {step_count} | "
                            f"Total Loss: {loss_dict['total']:.6f} | "
                            f"MSE: {loss_dict['mse']:.6f} | "
                            f"SSIM: {loss_dict['ssim']:.6f} | "
                            f"Boundary: {loss_dict.get('boundary', 0):.6f} | "
                            f"Smoothness: {loss_dict.get('smoothness', 0):.6f}"
                        )
                        
                        batch_frames_np = batch_frames.detach().cpu().numpy()
                        batch_heatmaps_np = batch_heatmaps.detach().cpu().numpy()
                        pred_heatmaps_np = pred_heatmap.detach().cpu().numpy()

                        # Enhanced logging with loss components
                        log_dict = {
                            "episode_id": episode_id,
                            "timestep": step_count,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            **loss_dict,  # Log all loss components
                            "gt_heatmap_batch": [wandb.Image(h.squeeze()*255) for h in batch_heatmaps_np[:4]],
                            "pred_heatmap_batch": [wandb.Image(h.squeeze()*255) for h in pred_heatmaps_np[:4]],
                        }
                        wandb.log(log_dict)
                    

                
                # for current_timestep in range(len(frame_paths)):
                #     frame_np = np.array(Image.open(frame_paths[current_timestep])).astype(np.float32) / 255.0
                #     # Convert to tensor and add batch/channel dimensions if needed
                #     frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                #     frame_tensor = frame_tensor.to("cuda" if torch.cuda.is_available() else "cpu")

                #     gt_heatmap = np.array(Image.open(norm_frame_paths[current_timestep])).astype(np.float32) / 255.0
                #     #checksum
                #     if np.isclose(gt_heatmap.min(), gt_heatmap.max()):
                #         logger.warning(f"Skipping timestep {current_timestep} in episode {episode_id}: GT heatmap is constant (min == max == {gt_heatmap.min()})")
                #         continue

                #     gt_heatmap_tensor = torch.from_numpy(gt_heatmap).unsqueeze(0).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")  # (1, 1, H, W)

                    # pred_heatmap = model.forward(
                    #     instruction=instruction,
                    #     image_sequence=frame_paths,
                    #     heatmap_sequence=norm_frame_paths,
                    #     current_idx=current_timestep,
                    #     history_length=0
                    # )
                    # pred_heatmap= model(frame_tensor, [instruction])
                    
                    

                    # if pred_heatmap.shape[-2:] != gt_heatmap_tensor.shape[-2:]:
                    #     gt_heatmap_tensor = F.interpolate(gt_heatmap_tensor, size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=True)

                    # mse_loss = F.mse_loss(pred_heatmap.float(), gt_heatmap_tensor.float())
                    # ssim_value = ssim(pred_heatmap, gt_heatmap_tensor, data_range=1.0, size_average=True)
                    # ssim_loss = 1 - ssim_value  # SSIM is similarity, so 1-SSIM is the loss
                    # total_loss= 0.9*mse_loss+ 0.1*ssim_loss

                    # epoch_loss.append(total_loss)

                    # step_count+=1
                    # if step_count % ACCUMULATION_STEPS == 0:
                    #     loss= torch.stack(epoch_loss).mean()
                    #     loss.backward()
                    #     optimizer.step()
                    #     optimizer.zero_grad()
                    #     # mse_loss.backward()
                    #     # ssim_loss.backward()
                    #     # total_loss.backward()
                    #     epoch_loss= []

                    #     logger.info(f"Episode {episode_id} | Timestep {current_timestep} | Heatmap shape: {pred_heatmap.shape} | MSE Loss: {total_loss.item():.6f}")
                    #     wandb.log({
                    #         "episode_id": episode_id,
                    #         "timestep": current_timestep,
                    #         "total_loss": loss.item(),
                    #         "mse_loss": mse_loss.item(),
                    #         "ssim_loss": ssim_loss.item(),
                    #         "gt_heatmap": wandb.Image(gt_heatmap * 255),  # scale for visualization
                    #         "pred_heatmap": wandb.Image(pred_heatmap.detach().cpu().squeeze().numpy() * 255)
                    #     })
                

    # Step the scheduler at the end of each epoch
    if 'scheduler' in locals():
        scheduler.step()
    
    save_path = f"/data/ws/VLN-CE/chkpt/clipunet_continuous_ep{epoch}_final.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() else None,
    }, save_path)
    logger.info(f"Saved full checkpoint to {save_path}")
    wandb.finish()