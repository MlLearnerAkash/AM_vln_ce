import os
import glob
import json
import numpy as np
import h5py
from PIL import Image
from typing import NamedTuple
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from utils.h5_writer import load_episode_from_hdf5
from pycocotools import mask as mask_utils
import torch


class NodeEntry(NamedTuple):
    node_id:  int
    mask:     np.ndarray
    path_row: np.ndarray

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
            episode_dirs = [os.path.join(self.data_root, "episode_"+eid) for eid in episode_ids]
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

        # Filter empty masks (all-zero rows) before passing to model
        valid = masks.any(axis=(1, 2))           # [K] bool
        masks = masks[valid]
        class_ids = class_ids[valid]
        geodesic = geodesic[valid]
        K = masks.shape[0]                        # actual number of valid objects

        masks = torch.from_numpy(masks).bool()
        class_ids = torch.from_numpy(class_ids).long()
        geodesic = torch.from_numpy(geodesic).float()

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


# ---------------------------------------------------------------------------
# H5-backed dataset: RLE masks + PL scores + JSONL instructions
# ---------------------------------------------------------------------------

class H5MaskPLSDataset(Dataset):
    """
    Dataset that loads per-frame RGB images, instance-segmentation masks, and
    per-object path-length (PL) scores from an HDF5 file, paired with
    natural-language instructions supplied through a JSONL file.

    HDF5 structure
    --------------
    Top-level key  : "{ep_folder}_{frame_idx}"   e.g. "1S7LAXRdDqK_0000000_plant_42__0"
    Sub-datasets:
        img_masks/{i}  int64 1-D  – F-major alternating skip/set RLE for mask i
        img_pls        float64 [K] – per-object PL scores
        size           int64   [2] – [H, W] of the frame

    JSONL structure (one JSON object per line)
    ------------------------------------------
        {"filename": "<ep_folder_name>", "instruction": "natural language ..."}

    "filename" is matched to the episode-folder component of the H5 key.
    Trailing underscores are stripped on both sides before comparison, so
    "1S7LAXRdDqK_0000000_plant_42_" matches "1S7LAXRdDqK_0000000_plant_42_"
    (or the same string without the trailing underscore).

    Image path
    ----------
    {base_dir}/trajectories/{ep_folder}/images/{frame_idx:05d}.png
    """

    def __init__(
        self,
        h5_path: str,
        jsonl_path: str,
        base_dir: str,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        max_instruction_length: int = 77,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.traj_root = os.path.join(base_dir, "trajectories")
        self.max_instruction_length = max_instruction_length
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # H5 handle – opened lazily in each DataLoader worker (fork-safe).
        self._h5 = None

        # Build instruction lookup: stripped_ep_folder -> instruction
        instruction_map: dict[str, str] = {}
        with open(jsonl_path, "r") as fj:
            for line in fj:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                instruction_map[entry["file"].removesuffix(".gif").rstrip("_")] = entry["instruction"]

        # Index all H5 keys that have a matching instruction entry.
        self.samples: list[dict] = []
        with h5py.File(h5_path, "r") as h5:
            for sample_key in h5.keys():
                # Split off the frame index from the right.
                ep_folder, frame_str = sample_key.rsplit("_", 1)
                lookup_key = ep_folder.rstrip("_")
                if lookup_key not in instruction_map:
                    continue
                self.samples.append({
                    "sample_key":  sample_key,
                    "ep_folder":   ep_folder,
                    "frame_idx":   int(frame_str),
                    "instruction": instruction_map[lookup_key],
                })

        print(
            f"[H5MaskPLSDataset] {len(self.samples)} samples from "
            f"{len(set(s['ep_folder'] for s in self.samples))} episodes"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _open_h5(self) -> "h5py.File":
        """Open the HDF5 file lazily (once per DataLoader worker process)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    @staticmethod
    def _decode_rle_mask(rle: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Decode a 1-D F-major alternating skip/set RLE into a (H, W) bool array.

        The RLE lists run lengths over a Fortran-order (column-major) flattened
        view of the image.  Runs at even positions (0, 2, …) are background
        (skip); runs at odd positions (1, 3, …) are foreground (set).
        """
        flat = np.zeros(H * W, dtype=bool)
        pos = 0
        for i, count in enumerate(rle):
            if i % 2 == 1:          # odd index → foreground run
                flat[pos: pos + count] = True
            pos += count
        return np.reshape(flat, (H, W), order="F")

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        h5   = self._open_h5()
        grp  = h5[meta["sample_key"]]

        H, W = int(grp["size"][0]), int(grp["size"][1])

        # Decode all masks from F-major RLE.
        masks_grp = grp["img_masks"]
        n_masks   = len(masks_grp)
        masks = np.stack(
            [self._decode_rle_mask(masks_grp[str(i)][()], H, W)
             for i in range(n_masks)],
            axis=0,
        )  # [K, H, W]  bool

        # PL scores – min-max normalized to [0, 1] per frame.
        pls = grp["img_pls"][()].astype(np.float32)  # [K]
        pls_min, pls_max = pls.min(), pls.max()
        if pls_max - pls_min > 1e-6:
            pls = (pls - pls_min) / (pls_max - pls_min)
        else:
            pls = np.zeros_like(pls)

        # RGB image.
        img_path = os.path.join(
            self.traj_root,
            meta["ep_folder"],
            "images",
            f"{meta['frame_idx']:05d}.png",
        )
        image       = Image.open(img_path).convert("RGB")
        clip_img    = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_img["pixel_values"].squeeze(0)  # [3, 224, 224]

        # Instruction tokens.
        clip_text = self.clip_processor(
            text=meta["instruction"],
            max_length=self.max_instruction_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids     = clip_text["input_ids"].squeeze(0)      # [L]
        attention_mask = clip_text["attention_mask"].squeeze(0) # [L]

        return {
            "pixel_values":  pixel_values,                    # [3, 224, 224]
            "masks":         torch.from_numpy(masks).bool(),  # [K, H, W]
            "pls":           torch.from_numpy(pls),           # [K]
            "input_ids":     input_ids,                       # [L]
            "attention_mask": attention_mask,                  # [L]
            "num_objects":   n_masks,
            "sample_key":    meta["sample_key"],
            "ep_folder":     meta["ep_folder"],
            "frame_idx":     meta["frame_idx"],
        }


def h5_maskpls_collate_fn(batch: list[dict]) -> dict:
    """
    Collate for H5MaskPLSDataset.

    Fixed-size tensors are stacked; variable-size masks and pls are kept as
    lists (one element per sample in the batch).
    """
    return {
        "pixel_values":    torch.stack([b["pixel_values"]   for b in batch]),
        "input_ids":       torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask":  torch.stack([b["attention_mask"] for b in batch]),
        "masks_list":      [b["masks"] for b in batch],
        "pls_list":        [b["pls"]   for b in batch],
        "num_objects":     [b["num_objects"] for b in batch],
        "sample_keys":     [b["sample_key"]  for b in batch],
    }


def create_h5_maskpls_dataloader(
    h5_path: str,
    jsonl_path: str,
    base_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    clip_model: str = "openai/clip-vit-base-patch16",
) -> DataLoader:
    """
    Convenience factory that returns a single DataLoader backed by
    H5MaskPLSDataset.

    Parameters
    ----------
    h5_path    : path to the HDF5 file containing masks and PL scores.
    jsonl_path : path to the JSONL file with "filename" / "instruction" pairs.
    base_dir   : root directory that contains the ``trajectories/`` sub-folder.
    batch_size : samples per batch.
    shuffle    : whether to shuffle the dataset each epoch.
    num_workers: number of DataLoader worker processes.
    clip_model : HuggingFace model name used for image/text preprocessing.
    """
    ds = H5MaskPLSDataset(
        h5_path=h5_path,
        jsonl_path=jsonl_path,
        base_dir=base_dir,
        clip_model_name=clip_model,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=h5_maskpls_collate_fn,
        pin_memory=True,
        drop_last=shuffle,   # drop incomplete batches only during training
    )


class H5EpisodePathLengthsDataset(Dataset):
    def __init__(self, h5_path: str, episode_ids: list = None):
        super().__init__()
        self.h5_path= h5_path
        self.clip_processor= CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self._cached_ep_id: str = None
        self._cached_ep_data: dict= None

        with h5py.File(h5_path, 'r') as hf:
            all_keys = sorted(hf.keys())
        self.episode_ids = episode_ids if episode_ids is not None else all_keys
        #all_keys#episode_ids if episode_ids is not None else all_keys

        self._ep_start = []
        self._fr_local = []
        self.episode_frame_counts = {}

        for ep in self.episode_ids:
            try:
                ep_data= load_episode_from_hdf5(self.h5_path, ep)
            except Exception:
                continue
            n_frames = len(ep_data['frame_data'])
            self.episode_frame_counts[ep] = n_frames

            for local_i, fd in enumerate(ep_data["frame_data"]):
                self._ep_start.append(ep)
                self._fr_local.append(local_i)

    def __len__(self):
        return len(self._ep_start)
    
    def _get_episode(self, ep_id: str) -> dict:
        """Load episode only when it changes (cache the current one)."""
        if self._cached_ep_id != ep_id:
            self._cached_ep_data = load_episode_from_hdf5(self.h5_path, ep_id)
            self._cached_ep_id = ep_id
        return self._cached_ep_data
    
    def __getitem__(self, idx):
        ep_id= self._ep_start[idx]
        local_frame_idx = self._fr_local[idx]
        ep_data = self._get_episode(ep_id)

        G = ep_data['graph']
        fd = ep_data['frame_data'][local_frame_idx]
        frame_idx = int(fd['frame_idx'])

        all_paths= G.graph.get("all_paths_lengths", None)
        if all_paths is None:
            raise KeyError(f"Episode {ep_id} missing 'all_paths_lengths'")

        all_nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        frame_node_ids = [n for n in all_nodes if G.nodes[n]['map'][0] == frame_idx]
        indices = [node_to_idx[n] for n in frame_node_ids]

        if indices:
            path_rows = all_paths[np.ix_(indices, indices)].astype(np.float32)
        else:
            path_rows = np.zeros((0, 0), dtype=np.float32)

        rgb = fd['rgb']
        H, W = rgb.shape[:2]

        masks_list = []
        for n in frame_node_ids:
            rle = G.nodes[n].get('segmentation')
            if rle is None:
                masks_list.append(np.zeros((H, W), dtype=bool))
                continue
            try:
                if 'size' not in rle:
                    rle = {'size': [H, W], 'counts': rle['counts']}
                comp = mask_utils.frPyObjects(rle, rle['size'][0], rle['size'][1])
                dec = mask_utils.decode(comp)
                m = (dec[..., 0] if dec.ndim == 3 else dec).astype(bool)
            except Exception:
                m = np.zeros((H, W), dtype=bool)
            masks_list.append(m)

        masks_arr = np.stack(masks_list) if masks_list else np.zeros((0, H, W), dtype=bool)
        node_registry = {
        node_id: NodeEntry(
            node_id  = node_id,
            mask     = masks_arr[k],
            path_row = path_rows[k],
        )
        for k, node_id in enumerate(frame_node_ids)
        }

        clip_text = self.clip_processor(
        text=ep_data.get('instruction', ''),
        padding='max_length', truncation=True,
        max_length=77, return_tensors='pt',
        )

        return {
        'episode_id':     ep_id,
        'frame_idx':      frame_idx,
        'frame_rgb':      rgb,
        "node_registry":  node_registry,
        "frame_node_ids":  frame_node_ids,
        'input_ids':      clip_text['input_ids'].squeeze(0),
        'attention_mask': clip_text['attention_mask'].squeeze(0),
        'node_to_idx':    node_to_idx, 
    }


def create_h5_episode_pathlengths_dataloader(h5_path: str, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0, val_split: float= 0.2, seed: int = 42,):
    with h5py.File(h5_path, 'r') as hf:
        all_keys = sorted(hf.keys())

    # Shuffle and split at episode level
    rng = random.Random(seed)
    keys = list(all_keys)
    rng.shuffle(keys)

    split_idx = int(len(keys) * (1 - val_split))
    train_ids = keys[:split_idx]
    val_ids   = keys[split_idx:]

    train_ds = H5EpisodePathLengthsDataset(h5_path, episode_ids=train_ids)
    val_ds= H5EpisodePathLengthsDataset(h5_path, episode_ids= val_ids)
    def h5_episode_pathlengths_collate_fn(batch: list[dict]) -> dict:
        """Collate: stack token tensors, keep variable-length masks/path_rows as lists."""
        return {
            "input_ids":       torch.stack([b["input_ids"] for b in batch]),
            "attention_mask":  torch.stack([b["attention_mask"] for b in batch]),
            "frame_rgbs":      [b["frame_rgb"] for b in batch],
            "episode_ids":     [b["episode_id"] for b in batch],
            "frame_idxs":      [b["frame_idx"] for b in batch],
            "node_registries":  [b["node_registry"] for b in batch],
            "node_to_idx":      [b["node_to_idx"] for b in batch],
            "frame_node_ids":   [b["frame_node_ids"] for b in batch],
        }
    train_loader= DataLoader(train_ds, batch_size=batch_size,shuffle=shuffle,
                             num_workers=num_workers,collate_fn=h5_episode_pathlengths_collate_fn,
                             pin_memory= True)
    val_loader= DataLoader(val_ds, batch_size=batch_size,shuffle=shuffle,
                             num_workers=num_workers,collate_fn=h5_episode_pathlengths_collate_fn,
                             pin_memory= True)
    return train_loader, val_loader
