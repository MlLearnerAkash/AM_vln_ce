import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EpisodeDataset(Dataset):
    def __init__(self, frames, heatmaps, instruction, semantic_maps=None, transform=None):
        """
        frames:        list of numpy arrays (H, W, 3)  — RGB observations
        heatmaps:      list of numpy arrays (H, W)     — normalised geodesic distance GT
        instruction:   string (same for all steps)
        semantic_maps: list of numpy arrays (H, W)     — integer object-ID maps.
                       If None, a zero map is used (backward compatibility).
        transform:     optional callable applied to (frame, heatmap, semantic_map)
        """
        self.frames       = frames
        self.heatmaps     = heatmaps
        self.instruction  = instruction
        self.semantic_maps = semantic_maps  # may be None
        self.transform    = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame   = self.frames[idx]    # (H, W, 3) uint8
        heatmap = self.heatmaps[idx]  # (H, W)    uint8 or float32

        if self.semantic_maps is not None:
            sem = self.semantic_maps[idx].copy()  # (H, W) int or float
        else:
            sem = np.zeros(frame.shape[:2], dtype=np.float32)

        if self.transform:
            frame, heatmap, sem = self.transform(frame, heatmap, sem)

        # RGB image → float32 [0, 1], shape (3, H, W)
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Geodesic heatmap → float32 [0, 1], shape (1, H, W)
        if heatmap.dtype == np.uint8:
            heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).float() / 255.0
        else:
            heatmap_t = torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0)

        # Semantic map → float32, shape (1, H, W).
        # Normalise object IDs to [0, 1] so they form a meaningful input channel.
        sem = sem.astype(np.float32)
        sem_max = max(sem.max(), 1.0)  # avoid div-by-zero if only background
        sem_norm = sem / sem_max       # [0, 1]
        sem_t = torch.from_numpy(sem_norm).unsqueeze(0)  # (1, H, W)

        # Raw integer semantic map for loss masking, shape (1, H, W)
        sem_ids = torch.from_numpy(sem.astype(np.int32)).unsqueeze(0)  # (1, H, W) int

        return frame_t, heatmap_t, sem_t, sem_ids, self.instruction


def collate_fn(batch):
    frames, heatmaps, sem_norms, sem_ids, instructions = zip(*batch)
    frames    = torch.stack(frames)    # (B, 3, H, W)
    heatmaps  = torch.stack(heatmaps)  # (B, 1, H, W)
    sem_norms = torch.stack(sem_norms) # (B, 1, H, W)  normalised — used as 4th input channel
    sem_ids   = torch.stack(sem_ids)   # (B, 1, H, W)  raw ids   — used in ordinal loss
    # All instructions are the same within an episode; take the first
    return frames, heatmaps, sem_norms, sem_ids, instructions[0]
