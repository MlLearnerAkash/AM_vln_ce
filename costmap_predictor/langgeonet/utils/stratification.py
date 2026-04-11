#!/usr/bin/env python3
"""
stratification.py
=================
Per-episode stratification of an e3d-action HDF5 dataset.

Two independent caps are applied to each episode's frame set:

  1. **Forward cap** – frames whose next_action is MOVE_FORWARD (direction =
     forward / straight / ahead) are down-sampled so they represent at most
     ``--forward-cap`` fraction of the kept frames  (default: 20 %).

  2. **Wall cap** – frames whose next_action_instruction names "wall" as the
     anchor object are down-sampled so they represent at most ``--wall-cap``
     fraction of the kept frames  (default: 10 %).

Frames are dropped at random inside each episode; a fixed seed makes the
result reproducible.  Temporal ordering of kept frames is preserved.
Episode-level data (graph, instruction) is copied unchanged.

Usage
-----
    python utils/stratification.py \\
        --in  /media/.../e3d_train.h5 \\
        --out /media/.../e3d_train_stratified.h5 \\
        [--forward-cap 0.20] \\
        [--wall-cap    0.10] \\
        [--seed        42]   \\
        [--dry-run]          # analyse only, do not write output
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Direction / object parsing  (mirrors analyse_e3d_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────
_DIR_LEFT    = re.compile(r'\bleft\b',                             re.IGNORECASE)
_DIR_RIGHT   = re.compile(r'\bright\b',                            re.IGNORECASE)
_DIR_FORWARD = re.compile(r'\b(forward|straight|ahead)\b',         re.IGNORECASE)
_DIR_STOP    = re.compile(r'\b(halt|stop|destination|arrived)\b',  re.IGNORECASE)

_OBJ_NAMES   = sorted([
    "void", "wall", "floor", "chair", "door", "table", "picture", "cabinet",
    "cushion", "window", "sofa", "bed", "curtain", "chest_of_drawers", "plant",
    "sink", "stairs", "ceiling", "toilet", "stool", "towel", "mirror",
    "tv_monitor", "shower", "column", "bathtub", "counter", "fireplace",
    "lighting", "beam", "railing", "shelving", "blinds", "gym_equipment",
    "seating", "board_panel", "furniture", "appliances", "clothes", "objects",
], key=len, reverse=True)

_OBJ_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(o) for o in _OBJ_NAMES) + r')\b', re.IGNORECASE
)


def _parse_instruction(text: str) -> tuple[str, str | None]:
    """Return (direction_label, object_name | None)."""
    if _DIR_LEFT.search(text):
        direction = "left"
    elif _DIR_RIGHT.search(text):
        direction = "right"
    elif _DIR_FORWARD.search(text):
        direction = "forward"
    elif _DIR_STOP.search(text):
        direction = "stop"
    else:
        direction = "other"
    m = _OBJ_PATTERN.search(text)
    obj = m.group(1).lower() if m else None
    return direction, obj


# ─────────────────────────────────────────────────────────────────────────────
# Core stratification logic
# ─────────────────────────────────────────────────────────────────────────────
def _stratify_episode(
    frame_metas: list[tuple[str, str, bool]],   # (fk, direction, is_wall)
    forward_cap: float,
    wall_cap: float,
    rng: np.random.Generator,
) -> tuple[list[str], dict]:
    """
    Return (sorted list of kept frame keys, stats_dict).

    Algorithm
    ---------
    Step 1 – Forward cap
      Keep ALL non-forward frames.
      Sample forward frames so that:
          n_fwd_kept / (n_fwd_kept + n_nonfwd) ≤ forward_cap
      ⟹  max_fwd = floor(forward_cap / (1 − forward_cap) × n_nonfwd)
      If n_nonfwd == 0, all frames are kept (can't cap relative to nothing).

    Step 2 – Wall cap  (applied to the set surviving step 1)
      From the surviving set keep ALL non-wall frames.
      Sample wall frames so that:
          n_wall_kept / (n_wall_kept + n_nowall) ≤ wall_cap
      If n_nowall == 0, all surviving wall frames are kept.
    """
    all_fwd    = [fk for fk, d, _ in frame_metas if d == "forward"]
    all_nonfwd = [fk for fk, d, _ in frame_metas if d != "forward"]
    is_wall    = {fk: w for fk, _, w in frame_metas}

    # ── Step 1: cap forward ───────────────────────────────────────────────
    n_nonfwd = len(all_nonfwd)
    dropped_fwd = 0

    if forward_cap < 1.0 and n_nonfwd > 0:
        max_fwd = int(np.floor(forward_cap / (1.0 - forward_cap) * n_nonfwd))
        if len(all_fwd) > max_fwd:
            dropped_fwd = len(all_fwd) - max_fwd
            kept_fwd = list(rng.choice(all_fwd, size=max_fwd, replace=False))
        else:
            kept_fwd = all_fwd
    else:
        kept_fwd = all_fwd

    after_step1 = all_nonfwd + kept_fwd

    # ── Step 2: cap wall (within the step-1 survivor set) ────────────────
    surviving_wall   = [fk for fk in after_step1 if is_wall[fk]]
    surviving_nowall = [fk for fk in after_step1 if not is_wall[fk]]
    n_nowall = len(surviving_nowall)
    dropped_wall = 0

    if wall_cap < 1.0 and n_nowall > 0:
        max_wall = int(np.floor(wall_cap / (1.0 - wall_cap) * n_nowall))
        if len(surviving_wall) > max_wall:
            dropped_wall = len(surviving_wall) - max_wall
            kept_wall = list(rng.choice(surviving_wall, size=max_wall, replace=False))
        else:
            kept_wall = surviving_wall
    else:
        kept_wall = surviving_wall

    kept = surviving_nowall + kept_wall
    kept_sorted = sorted(kept)   # preserve temporal order (frame keys are zero-padded ints)

    stats = {
        "n_orig":        len(frame_metas),
        "n_kept":        len(kept_sorted),
        "dropped_fwd":   dropped_fwd,
        "dropped_wall":  dropped_wall,
        "n_fwd_orig":    len(all_fwd),
        "n_fwd_kept":    len(kept_fwd),
        "n_wall_orig":   len([fk for fk, _, w in frame_metas if w]),
        "n_wall_kept":   len(kept_wall),
    }
    return kept_sorted, stats


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 copy helper
# ─────────────────────────────────────────────────────────────────────────────
def _copy_dataset(src_ds: h5py.Dataset, dst_grp: h5py.Group, name: str) -> None:
    """Copy one dataset preserving compression and chunk layout."""
    kwargs: dict = {}
    if src_ds.chunks is not None:
        kwargs["chunks"]      = src_ds.chunks
        kwargs["compression"] = src_ds.compression
        kwargs["compression_opts"] = src_ds.compression_opts
    dst_grp.create_dataset(name, data=src_ds[()], **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────────────────────
def stratify(
    in_path:     str,
    out_path:    str | None,
    forward_cap: float = 0.20,
    wall_cap:    float = 0.10,
    seed:        int   = 42,
    dry_run:     bool  = False,
) -> None:
    t0  = time.time()
    rng = np.random.default_rng(seed)

    sep  = "─" * 70
    sep2 = "═" * 70

    print(f"\n{sep2}")
    print("  Stratification")
    print(f"  Input  : {in_path}")
    if not dry_run:
        print(f"  Output : {out_path}")
    print(f"  Forward cap : {forward_cap*100:.0f}%   Wall cap : {wall_cap*100:.0f}%")
    print(f"  Seed        : {seed}   Dry-run : {dry_run}")
    print(f"{sep2}\n")

    # ── global accumulators ───────────────────────────────────────────────────
    g_orig = g_kept = g_drop_fwd = g_drop_wall = 0
    dir_before  = Counter()
    dir_after   = Counter()
    obj_before  = Counter()
    obj_after   = Counter()

    with h5py.File(in_path, "r") as hf_in:
        ep_keys = sorted(hf_in.keys())
        print(f"Episodes found : {len(ep_keys)}\n")

        out_ctx = (
            h5py.File(out_path, "w") if not dry_run
            else _NullFile()
        )

        with out_ctx as hf_out:
            for ep_i, ep_key in enumerate(ep_keys):
                ep_in      = hf_in[ep_key]
                frame_grps = ep_in["frames"]

                # ── parse per-frame metadata ─────────────────────────────
                frame_metas: list[tuple[str, str, bool]] = []
                for fk in sorted(frame_grps.keys()):
                    fg = frame_grps[fk]
                    if "next_action_instruction" in fg:
                        raw = fg["next_action_instruction"][()]
                        nai = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        direction, obj = _parse_instruction(nai)
                    else:
                        direction, obj = "other", None
                    is_wall = (obj == "wall")
                    frame_metas.append((fk, direction, is_wall))

                    dir_before[direction] += 1
                    if obj:
                        obj_before[obj] += 1

                # ── stratify ────────────────────────────────────────────
                kept_keys, stats = _stratify_episode(
                    frame_metas, forward_cap, wall_cap, rng
                )

                # Tally after stats
                kept_set  = set(kept_keys)
                meta_map  = {fk: (d, w) for fk, d, w in frame_metas}
                for fk in kept_keys:
                    d, w = meta_map[fk]
                    dir_after[d] += 1
                # count wall obj after
                fk_to_obj = {}
                for fg_key in sorted(frame_grps.keys()):
                    fg = frame_grps[fg_key]
                    if "next_action_instruction" in fg:
                        raw = fg["next_action_instruction"][()]
                        nai = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                        _, obj = _parse_instruction(nai)
                        fk_to_obj[fg_key] = obj
                for fk in kept_keys:
                    obj = fk_to_obj.get(fk)
                    if obj:
                        obj_after[obj] += 1

                g_orig      += stats["n_orig"]
                g_kept      += stats["n_kept"]
                g_drop_fwd  += stats["dropped_fwd"]
                g_drop_wall += stats["dropped_wall"]

                fwd_pct  = (stats["n_fwd_kept"]  / stats["n_kept"]  * 100) if stats["n_kept"]  else 0
                wall_pct = (stats["n_wall_kept"] / stats["n_kept"] * 100) if stats["n_kept"] else 0

                print(
                    f"  ep {ep_key:>6}  "
                    f"orig={stats['n_orig']:4d}  kept={stats['n_kept']:4d}  "
                    f"drop_fwd={stats['dropped_fwd']:4d}  drop_wall={stats['dropped_wall']:4d}  "
                    f"fwd={fwd_pct:4.1f}%  wall={wall_pct:4.1f}%"
                )

                # ── write output ─────────────────────────────────────────
                if not dry_run:
                    ep_out = hf_out.create_group(ep_key)

                    # Copy episode-level datasets (graph, instruction, …)
                    for key in ep_in.keys():
                        if key == "frames":
                            continue
                        hf_in.copy(ep_in[key], ep_out, name=key)

                    # Copy only kept frames
                    frames_out = ep_out.create_group("frames")
                    for fk in kept_keys:
                        hf_in.copy(frame_grps[fk], frames_out, name=fk)

    elapsed = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # Summary report
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  SUMMARY")
    print(sep2)
    print(f"  Total frames  before : {g_orig:>7,}")
    print(f"  Total frames  after  : {g_kept:>7,}  ({100*g_kept/max(1,g_orig):.1f}% retained)")
    print(f"  Dropped (forward cap): {g_drop_fwd:>7,}")
    print(f"  Dropped (wall cap)   : {g_drop_wall:>7,}")
    print()

    print(sep)
    print("  DIRECTIONAL DISTRIBUTION")
    print(sep)
    all_dirs = ["forward", "left", "right", "stop", "other"]
    tot_b = sum(dir_before.values())
    tot_a = sum(dir_after.values())
    print(f"  {'Direction':<12} {'Before':>8}  {'%':>6}    {'After':>8}  {'%':>6}")
    print(f"  {'-'*12} {'-'*8}  {'-'*6}    {'-'*8}  {'-'*6}")
    for d in all_dirs:
        b = dir_before.get(d, 0)
        a = dir_after.get(d, 0)
        print(f"  {d:<12} {b:>8,}  {100*b/max(1,tot_b):>5.1f}%    {a:>8,}  {100*a/max(1,tot_a):>5.1f}%")

    print()
    print(sep)
    print("  OBJECT ANCHOR DISTRIBUTION  (top 15)")
    print(sep)
    all_objs = sorted(set(list(obj_before.keys()) + list(obj_after.keys())),
                      key=lambda o: -obj_before.get(o, 0))
    tot_ob = sum(obj_before.values())
    tot_oa = sum(obj_after.values())
    print(f"  {'Object':<22} {'Before':>8}  {'%':>6}    {'After':>8}  {'%':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}    {'-'*8}  {'-'*6}")
    for obj in all_objs[:15]:
        b = obj_before.get(obj, 0)
        a = obj_after.get(obj, 0)
        print(f"  {obj:<22} {b:>8,}  {100*b/max(1,tot_ob):>5.1f}%    {a:>8,}  {100*a/max(1,tot_oa):>5.1f}%")

    print()
    print(f"  Wall fraction before : {100*obj_before.get('wall',0)/max(1,tot_ob):.1f}%")
    print(f"  Wall fraction after  : {100*obj_after.get('wall',0)/max(1,tot_oa):.1f}%")
    print()
    print(f"  Elapsed : {elapsed:.1f}s")
    if not dry_run:
        import os
        out_gb = os.path.getsize(out_path) / 1024**3
        print(f"  Output file size : {out_gb:.2f} GB  →  {out_path}")
    print(sep2 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Null context manager for dry-run mode
# ─────────────────────────────────────────────────────────────────────────────
class _NullFile:
    """Drop-in replacement for h5py.File that discards all writes."""
    def __enter__(self):           return self
    def __exit__(self, *_):        pass
    def create_group(self, _):     return _NullFile()
    def create_dataset(self, *a, **kw): pass
    def keys(self):                return iter([])
    def copy(self, *a, **kw):      pass


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in", dest="in_path",
        default="/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_test.h5",
        help="Input HDF5 file",
    )
    parser.add_argument(
        "--out", dest="out_path",
        default=None,
        help="Output HDF5 file (required unless --dry-run)",
    )
    parser.add_argument(
        "--forward-cap", type=float, default=0.20,
        help="Max fraction of kept frames that may be FORWARD (default: 0.20)",
    )
    parser.add_argument(
        "--wall-cap", type=float, default=0.10,
        help="Max fraction of kept frames that may have 'wall' as anchor (default: 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible frame sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse and print statistics only; do not write any output file",
    )
    args = parser.parse_args()

    if not args.dry_run and args.out_path is None:
        parser.error("--out is required unless --dry-run is set")

    if not args.dry_run and args.out_path == args.in_path:
        parser.error("--out must differ from --in (cannot overwrite input)")

    for cap_name, cap_val in [("--forward-cap", args.forward_cap),
                               ("--wall-cap",    args.wall_cap)]:
        if not (0.0 < cap_val <= 1.0):
            parser.error(f"{cap_name} must be in (0, 1]")

    stratify(
        in_path     = args.in_path,
        out_path    = args.out_path,
        forward_cap = args.forward_cap,
        wall_cap    = args.wall_cap,
        seed        = args.seed,
        dry_run     = args.dry_run,
    )
