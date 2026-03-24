import numpy as np
import matplotlib.pyplot as plt

def get_object_geodesic_distances(env, semantic_frame):
    """
    Returns a dictionary mapping object_id to geodesic distance from the object's 3D center
    to the goal position for the current episode.
    
    Args:
        env: The VLNCEDaggerEnv environment instance.
        semantic_frame: 2D numpy array of semantic object ids (obs["semantic"]).
    
    Returns:
        dict: {object_id: geodesic_distance_to_goal}
    """
    # Get goal position for the current episode
    goal_position = env.current_episode.goals[0].position

    # Get unique object ids in the semantic frame, skip background (-1)
    object_ids = [obj_id for obj_id in np.unique(semantic_frame) if obj_id != -1]

    distances = {}
    for obj_id in object_ids:
        try:
            obj = env._env.sim.semantic_annotations().objects[obj_id]
            obj_position = obj.aabb.center()  # 3D center of the object's bounding box

            geodesic_distance = env._env.sim.geodesic_distance(obj_position, goal_position)
            distances[obj_id] = 55 if geodesic_distance == np.inf else geodesic_distance
        except Exception as e:
            # Handle missing objects or errors gracefully
            distances[obj_id] = None
            print(f"Could not compute distance for object {obj_id}: {e}")
    return distances


def encode_normalized_distances_to_frame(semantic_frame, distances):
    """
    For each object in the semantic_frame, normalize its geodesic distance (from `distances`)
    to [0, 1] across all objects in the frame, and create a 2D array where each pixel's value
    is the normalized distance of its object.

    Args:
        semantic_frame (np.ndarray): 2D array of object ids (H, W).
        distances (dict): {object_id: geodesic_distance}

    Returns:
        np.ndarray: 2D array (H, W) with normalized geodesic distances per pixel.
    """
    # Filter out objects with None distances
    valid_distances = {k: v for k, v in distances.items() if v is not None}
    if not valid_distances:
        # If no valid distances, return zeros
        return np.zeros_like(semantic_frame, dtype=np.float32)

    # Get min and max for normalization
    dist_values = np.array(list(valid_distances.values()))
    min_dist, max_dist = dist_values.min(), dist_values.max()
    # Avoid division by zero
    if max_dist == min_dist:
        norm_distances = {k: 0.0 for k in valid_distances}
    else:
        norm_distances = {k: (v - min_dist) / (max_dist - min_dist) for k, v in valid_distances.items()}

    # Prepare output frame
    norm_frame = np.zeros_like(semantic_frame, dtype=np.float32)
    for obj_id, norm_val in norm_distances.items():
        mask = (semantic_frame == obj_id)
        norm_frame[mask] = norm_val

    # Optionally, set background (-1) to 0 or np.nan
    norm_frame[semantic_frame == -1] = 0.0

    return norm_frame

def encode_directional_cue_to_frame(semantic_frame, next_action):
    """
    Apply a directional spatial weight map to a semantic segmentation frame.

    The weight map encodes the intended navigation direction as a spatial pattern:

    - TURN_LEFT  (action 2): half-period sin wave, LEAST weight on the leftmost
                              column (objects you're turning away from) and
                              MOST weight on the rightmost column (objects that
                              will be in front after turning).
                              weight = sin(π/2 · x_norm)  → 0.0 at left, 1.0 at right.
    - TURN_RIGHT (action 3): mirror  →  weight = sin(π/2 · (1 - x_norm))
                              → 1.0 at left, 0.0 at right (leftmost gets most,
                              rightmost — where you turn away from — gets least).
    - MOVE_FORWARD / other : |tan(θ)| where θ is the horizontal angle from the
                              image centre, mapped over [−89°, +89°].  The
                              result is lowest (≈ 0) at the central column and
                              highest (= 1) at both edges, highlighting the
                              straight-ahead corridor as the dark region.

    All weight patterns are broadcast uniformly across every row (H dimension)
    so the output is a pure horizontal cue.

    Parameters
    ----------
    semantic_frame : np.ndarray, shape (H, W), int
        The *previous* semantic segmentation frame (the observation the agent
        had before taking next_action).
    next_action : int
        Navigation action token.  Expected values::
            0 – STOP
            1 – MOVE_FORWARD
            2 – TURN_LEFT
            3 – TURN_RIGHT

    Returns
    -------
    weight_map : np.ndarray, shape (H, W), float32
        Directional weight in [0, 1] for every pixel.
    cue_frame : np.ndarray, shape (H, W, 3), uint8
        Semantic frame rendered with a 'hot' colourmap and the weight map
        multiplied in, giving a coloured directional cue overlay.
    """
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    H, W = semantic_frame.shape

    x_norm = np.linspace(0.0, 1.0, W, dtype=np.float32)   # 0 = left, 1 = right

    if next_action == HabitatSimActions.TURN_LEFT:
        # Leftmost objects get the LEAST weight (turning away from left side).
        # Sin quarter-wave: 0.0 at left edge, rising to 1.0 at right edge.
        weights_1d = np.sin(np.pi / 2.0 * x_norm)                   # [W]

    elif next_action == HabitatSimActions.TURN_RIGHT:
        # Rightmost objects get the LEAST weight (turning away from right side).
        # Sin quarter-wave: 1.0 at left edge, tapering to 0.0 at right edge.
        weights_1d = np.sin(np.pi / 2.0 * (1.0 - x_norm))          # [W]

    else:
        # |tan(θ)| for θ in (−89°, +89°) centred on the image.
        # Gives ≈ 0 at the central column (straight-ahead) and 1 at both edges.
        eps = np.deg2rad(1.0)                                        # avoid tan(±90°)
        angles = np.linspace(-np.pi / 2.0 + eps, np.pi / 2.0 - eps, W, dtype=np.float32)
        abs_tan = np.abs(np.tan(angles))                             # [W]  0 → ∞ → 0
        max_val = abs_tan.max()
        weights_1d = abs_tan / max_val if max_val > 0 else abs_tan  # normalise to [0,1]

    # Broadcast 1-D weights to (H, W)
    weight_map = np.tile(weights_1d, (H, 1)).astype(np.float32)     # [H, W]

    segment_weight_map = np.zeros((H, W), dtype=np.float32)

    unique_ids = np.unique(semantic_frame)
    segment_weights = {}
    for seg_id in unique_ids:
        mask = semantic_frame == seg_id                              # [H, W] bool

        # Bounding-box centre of the segment
        rows, cols = np.where(mask)
        centre_row = int((rows.min() + rows.max()) / 2)
        centre_col = int((cols.min() + cols.max()) / 2)

        # Single weight value from the weight map at the centre pixel
        seg_weight = weight_map[centre_row, centre_col]
        segment_weights[int(seg_id)] = float(weight_map[centre_row, centre_col])
        # Assign this single value to all pixels belonging to this segment
        segment_weight_map[mask] = seg_weight

    # ── Coloured overlay ─────────────────────────────────────────────────────
    # Normalise semantic IDs to [0, 1] for colourmap indexing
    sem = semantic_frame.astype(np.float32)
    sem_min, sem_max = sem.min(), sem.max()
    if sem_max > sem_min:
        sem_norm = (sem - sem_min) / (sem_max - sem_min)
    else:
        sem_norm = np.zeros_like(sem)

    # Apply 'hot' colourmap: output (H, W, 4) RGBA in [0,1]
    import matplotlib.cm as cm
    colored = cm.hot(sem_norm)[..., :3]                              # [H, W, 3] float64

    # Multiply each channel by the weight map
    cue_frame = (colored * weight_map[..., None])
    cue_frame = np.clip(cue_frame * 255.0, 0, 255).astype(np.uint8) # [H, W, 3] uint8

    return segment_weights, cue_frame #segment_weight_map


def save_norm_frame_heatmap(norm_frame, save_path, cmap='viridis', upscale_factor=4):
    """
    Save the normalized frame as a heatmap image.

    Args:
        norm_frame (np.ndarray): 2D array of normalized values.
        save_path (str): Path to save the heatmap image.
        cmap (str): Matplotlib colormap.
        upscale_factor (int): Factor to upscale the image for higher resolution.
    """
    # Optional: upscale the frame for higher resolution visualization
    if upscale_factor > 1:
        norm_frame = np.kron(norm_frame, np.ones((upscale_factor, upscale_factor)))

    plt.figure(figsize=(norm_frame.shape[1]/10, norm_frame.shape[0]/10), dpi=10)
    plt.axis('off')
    plt.imshow(norm_frame, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label= "Distances(m)")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()