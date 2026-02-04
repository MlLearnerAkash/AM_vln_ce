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