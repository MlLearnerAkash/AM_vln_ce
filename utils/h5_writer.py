# """Streaming HDF5 writer for episodic datasets.

# Provides an API to create episode groups and append frames and metadata
# on-the-fly while episodes are being generated.

# Example usage:
#     from utils.h5_writer import H5EpisodeWriter

#     with H5EpisodeWriter('/path/to/out.h5') as writer:
#         writer.start_episode(42, instruction='go to the kitchen')
#         writer.add_frame(0, rgb_array)
#         writer.add_frame(1, rgb_array2)
#         writer.write_episode_graph(graph)
#         writer.finish_episode()

# The writer writes per-frame datasets under `/episode_<id>/rgb/frame_###` and
# stores `instruction` and `episode_graph_pickle` as vlen-bytes datasets.
# """

# import pickle
# import h5py
# import numpy as np
# from typing import Optional


# class H5EpisodeWriter:
#     def __init__(self, h5_path: str, compress: bool = True):
#         self.h5_path = h5_path
#         self.compress = compress
#         self._hf = None
#         self._current_ep = None

#     def __enter__(self):
#         # open file in append mode
#         self._hf = h5py.File(self.h5_path, 'a')
#         return self

#     def __exit__(self, exc_type, exc, tb):
#         try:
#             if self._hf is not None:
#                 self._hf.flush()
#                 self._hf.close()
#         finally:
#             self._hf = None

#     def start_episode(self, episode_id: int, instruction: Optional[str] = None):
#         if self._hf is None:
#             raise RuntimeError('HDF5 file not opened; use `with H5EpisodeWriter(...) as w:`')
#         ep_name = f'episode_{episode_id}'
#         if ep_name in self._hf:
#             # start fresh: remove existing group
#             del self._hf[ep_name]
#         grp = self._hf.create_group(ep_name)
#         # create rgb subgroup
#         grp.create_group('rgb')
#         # store instruction if provided
#         if instruction is not None:
#             dt = h5py.special_dtype(vlen=bytes)
#             grp.create_dataset('instruction', data=instruction.encode('utf-8'), dtype=dt)
#         self._current_ep = ep_name
#         self._hf.flush()

#     def add_frame(self, frame_idx: int, rgb: np.ndarray):
#         """Add a single RGB frame (HxWx3 uint8) to the current episode.

#         The dataset name will be `frame_{frame_idx:03d}` under the `rgb` subgroup.
#         """
#         if self._hf is None or self._current_ep is None:
#             raise RuntimeError('No active episode. Call start_episode first.')
#         ep_grp = self._hf[self._current_ep]
#         rgb_grp = ep_grp['rgb']
#         ds_name = f'frame_{frame_idx:03d}'
#         # Ensure dtype
#         arr = np.asarray(rgb, dtype=np.uint8)
#         if self.compress:
#             rgb_grp.create_dataset(ds_name, data=arr, compression='gzip', compression_opts=4)
#         else:
#             rgb_grp.create_dataset(ds_name, data=arr)
#         self._hf.flush()

#     def write_episode_graph(self, graph_obj):
#         """Pickle and store the episode graph object under the current episode."""
#         if self._hf is None or self._current_ep is None:
#             raise RuntimeError('No active episode. Call start_episode first.')
#         ep_grp = self._hf[self._current_ep]
#         graph_bytes = pickle.dumps(graph_obj)
#         dt = h5py.special_dtype(vlen=bytes)
#         # overwrite if exists
#         if 'episode_graph_pickle' in ep_grp:
#             del ep_grp['episode_graph_pickle']
#         ep_grp.create_dataset('episode_graph_pickle', data=graph_bytes, dtype=dt)
#         self._hf.flush()

#     def finish_episode(self):
#         """Finalize the episode (writes attrs like `num_frames`)."""
#         if self._hf is None or self._current_ep is None:
#             return
#         ep_grp = self._hf[self._current_ep]
#         rgb_grp = ep_grp['rgb']
#         num_frames = len(rgb_grp.keys())
#         ep_grp.attrs['num_frames'] = num_frames
#         # zero out current
#         self._current_ep = None
#         self._hf.flush()

import h5py
import numpy as np
import pickle
import io

def save_episode_to_hdf5(hdf5_path: str, ep_data: dict) -> None:
    """
    Append one episode into an HDF5 file.

    HDF5 layout
    -----------
    /<episode_id>/
        instruction          (scalar string dataset)
        graph                (raw bytes dataset — pickled networkx graph)
        frames/
            <000>/
                rgb          (H x W x 3  uint8)
                frame_idx    (scalar int)
            <001>/
                ...
    """
    episode_id = str(ep_data["episode_id"])

    with h5py.File(hdf5_path, "a") as hf:          # "a" = append / create
        if episode_id in hf:
            del hf[episode_id]                      # overwrite if re-running

        ep_grp = hf.create_group(episode_id)

        # ── instruction ───────────────────────────────────────────────────
        ep_grp.create_dataset(
            "instruction",
            data=ep_data["instruction"],            # h5py stores str natively
        )

        # ── episode graph (pickle → bytes → HDF5 opaque blob) ────────────
        buf = io.BytesIO()
        pickle.dump(ep_data["graph"], buf)
        graph_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        ep_grp.create_dataset(
            "graph",
            data=graph_bytes,
            compression="gzip",
            compression_opts=4,
        )

        # ── per-frame RGB images ──────────────────────────────────────────
        frames_grp = ep_grp.create_group("frames")

        for fd in ep_data["frame_data"]:
            frame_key = f"{fd['frame_idx']:03d}"
            frame_grp = frames_grp.create_group(frame_key)

            rgb = np.asarray(fd["rgb"], dtype=np.uint8)   # ensure ndarray
            frame_grp.create_dataset(
                "rgb",
                data=rgb,
                compression="gzip",
                compression_opts=4,
            )
            frame_grp.create_dataset(
                "frame_idx",
                data=fd["frame_idx"],
            )

def load_episode_from_hdf5(hdf5_path: str, episode_id: str) -> dict:
    """
    Read one episode back from the HDF5 file.

    Returns a dict with keys:
        episode_id  : str
        instruction : str
        graph       : original Python object (unpickled)
        frame_data  : list of {"frame_idx": int, "rgb": np.ndarray}
    """
    episode_id = str(episode_id)

    with h5py.File(hdf5_path, "r") as hf:
        if episode_id not in hf:
            raise KeyError(f"Episode '{episode_id}' not found in {hdf5_path}")

        ep_grp = hf[episode_id]

        instruction = ep_grp["instruction"][()].decode("utf-8") \
            if isinstance(ep_grp["instruction"][()], bytes) \
            else str(ep_grp["instruction"][()])

        graph_bytes = ep_grp["graph"][()].tobytes()
        graph = pickle.loads(graph_bytes)

        frame_data = []
        frames_grp = ep_grp["frames"]
        for frame_key in sorted(frames_grp.keys()):
            fg = frames_grp[frame_key]
            frame_data.append({
                "frame_idx": int(fg["frame_idx"][()]),
                "rgb":       fg["rgb"][()],             # np.ndarray uint8
            })

    return {
        "episode_id":  episode_id,
        "instruction": instruction,
        "graph":       graph,
        "frame_data":  frame_data,
    }