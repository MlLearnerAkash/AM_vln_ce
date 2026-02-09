import torch
from torch.utils.data import Dataset, DataLoader

class EpisodeDataset(Dataset):
    def __init__(self, frames, heatmaps, instruction, transform=None):
        """
        frames: list of numpy arrays (H, W, 3)
        heatmaps: list of numpy arrays (H, W)
        instruction: string (same for all)
        transform: optional transform to apply to frames/heatmaps
        """
        self.frames = frames
        self.heatmaps = heatmaps
        self.instruction = instruction
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        heatmap = self.heatmaps[idx]
        if self.transform:
            frame, heatmap = self.transform(frame, heatmap)
        # Convert to torch tensors
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        heatmap = torch.from_numpy(heatmap).unsqueeze(0).float() / 255.0  # (1, H, W)
        return frame, heatmap, self.instruction

def collate_fn(batch):
    frames, heatmaps, instructions = zip(*batch)
    frames = torch.stack(frames)
    heatmaps = torch.stack(heatmaps)
    # All instructions are the same, so just return the first
    return frames, heatmaps, instructions[0]
