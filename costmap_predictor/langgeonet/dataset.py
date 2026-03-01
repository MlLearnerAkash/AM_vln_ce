"""
Dataset for LangGeoNet training.

Each episode contains:
    - Multiple RGB frames (the trajectory)
    - One language instruction (shared across all frames in the episode)
    - Per-frame instance segmentation masks + class IDs
    - Per-object normalized geodesic distance to goal (ground truth)

Expected directory structure:
    data_root/
        train.txt / val.txt          # episode IDs, one per line
        episode_000/
            instruction.txt           # language instruction
            frame_000/
                rgb.png               # RGB image
                masks.npy             # [K, H, W] binary masks
                class_ids.npy         # [K] integer class IDs
                geodesic_distances.npy # [K] normalized geodesic distances
            frame_001/
                ...
        episode_001/
            ...
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor


class LangGeoNetDataset(Dataset):
    """
    Dataset that yields individual (frame, instruction, GT geodesic) tuples.
    The same instruction is shared by all frames within an episode.
    """

    def __init__(
        self,
        data_root,
        clip_model_name="openai/clip-vit-base-patch16",
        max_instruction_length=77,
        max_objects=50,
        split="train",
    ):
        super().__init__()
        self.data_root = data_root
        self.max_objects = max_objects
        self.max_instruction_length = max_instruction_length

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.samples = []
        self._build_index(split)

    def _build_index(self, split):
        """Index all (episode, frame) pairs."""
        split_file = os.path.join(self.data_root, f"{split}.txt")

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                episode_ids = [line.strip() for line in f if line.strip()]
            episode_dirs = [os.path.join(self.data_root, eid) for eid in episode_ids]
        else:
            episode_dirs = sorted(glob.glob(os.path.join(self.data_root, "episode_*")))

        for ep_dir in episode_dirs:
            if not os.path.isdir(ep_dir):
                continue

            instr_path = os.path.join(ep_dir, "instruction.txt")
            if not os.path.exists(instr_path):
                continue

            frame_dirs = sorted(glob.glob(os.path.join(ep_dir, "frame_*")))
            for frame_dir in frame_dirs:
                required = [
                    os.path.join(frame_dir, "rgb.png"),
                    os.path.join(frame_dir, "masks.npy"),
                    os.path.join(frame_dir, "class_ids.npy"),
                    os.path.join(frame_dir, "geodesic_distances.npy"),
                ]
                if all(os.path.exists(p) for p in required):
                    self.samples.append({
                        "episode_dir": ep_dir,
                        "frame_dir": frame_dir,
                        "instruction_path": instr_path,
                    })

        print(f"[LangGeoNetDataset] {len(self.samples)} frames from "
              f"{len(set(s['episode_dir'] for s in self.samples))} episodes ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -- Instruction --
        with open(sample["instruction_path"], "r") as f:
            instruction = f.read().strip()

        clip_text = self.clip_processor(
            text=instruction,
            max_length=self.max_instruction_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = clip_text["input_ids"].squeeze(0)
        attention_mask = clip_text["attention_mask"].squeeze(0)

        # -- RGB frame --
        image = Image.open(
            os.path.join(sample["frame_dir"], "rgb.png")
        ).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].squeeze(0)

        # -- Masks, classes, GT geodesic --
        masks = np.load(os.path.join(sample["frame_dir"], "masks.npy"))
        class_ids = np.load(os.path.join(sample["frame_dir"], "class_ids.npy"))
        geodesic = np.load(os.path.join(sample["frame_dir"], "geodesic_distances.npy"))

        K = min(masks.shape[0], self.max_objects)
        masks = torch.from_numpy(masks[:K]).bool()
        class_ids = torch.from_numpy(class_ids[:K]).long()
        geodesic = torch.from_numpy(geodesic[:K]).float()

        return {
            "pixel_values": pixel_values,         # [3, 224, 224]
            "masks": masks,                        # [K, H, W]
            "class_ids": class_ids,                # [K]
            "input_ids": input_ids,                # [L]
            "attention_mask": attention_mask,       # [L]
            "geodesic_distances": geodesic,        # [K]
            "num_objects": K,
            "episode_dir": sample["episode_dir"],
            "frame_dir": sample["frame_dir"],
        }


def langgeonet_collate_fn(batch):
    """
    Custom collate: stack fixed-size tensors, keep variable-size as lists.
    """
    return {
        "pixel_values":       torch.stack([b["pixel_values"] for b in batch]),
        "input_ids":          torch.stack([b["input_ids"] for b in batch]),
        "attention_mask":     torch.stack([b["attention_mask"] for b in batch]),
        "masks_list":         [b["masks"] for b in batch],
        "class_ids_list":     [b["class_ids"] for b in batch],
        "geodesic_dists_list":[b["geodesic_distances"] for b in batch],
        "num_objects":        [b["num_objects"] for b in batch],
    }


def create_dataloaders(data_root, batch_size=8, num_workers=4,
                       clip_model="openai/clip-vit-base-patch16"):
    """Create train and val dataloaders."""
    train_ds = LangGeoNetDataset(data_root, clip_model, split="train")
    val_ds   = LangGeoNetDataset(data_root, clip_model, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=langgeonet_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=langgeonet_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader
