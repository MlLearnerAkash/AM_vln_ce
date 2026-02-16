# Semantic Sensor Configuration - Complete Fix Guide

## Problems Identified

Your semantic sensor pipeline had **3 critical configuration issues**:

### 1. **Missing SCENE_DATASET in Default Config**
- The Habitat-Lab default config (`habitat/config/default.py`) had no `SCENE_DATASET` configuration option
- Even though you added `SCENE_DATASET` to the task YAML, it had nowhere to be defined in the default schema

### 2. **SCENE_DATASET Not Passed to Simulator**
- The `HabitatSim.create_sim_config()` method was only setting `sim_cfg.scene_id` but **NOT** setting `sim_cfg.scene_dataset_config_file`
- This meant the semantic annotations configuration was never passed to habitat-sim, even though you specified it in the YAML

### 3. **Episode Generation Issue**
- The `episode_generator.py` calls `get_object_geodesic_distances()` which needs properly loaded semantic annotations
- If semantic annotations weren't loaded due to issue #2, geodesic distances would fail or return same values

## Solutions Applied

### ✅ Fix 1: Added SCENE_DATASET to Default Config
**File**: [habitat-lab/habitat/config/default.py](habitat-lab/habitat/config/default.py#L216)

```python
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.SIMULATOR.SCENE_DATASET = ""  # ← ADDED THIS LINE
_C.SIMULATOR.SEED = _C.SEED
```

### ✅ Fix 2: Updated HabitatSim to Use SCENE_DATASET
**File**: [habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py](habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py#L777-L780)

```python
# 1. Simulator Configuration
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = self.habitat_config.SCENE
if hasattr(self.habitat_config, "SCENE_DATASET") and self.habitat_config.SCENE_DATASET:
    sim_cfg.scene_dataset_config_file = self.habitat_config.SCENE_DATASET  # ← ADDED THIS
sim_cfg.gpu_device_id = self.habitat_config.HABITAT_SIM_V0.GPU_DEVICE_ID
```

### ✅ Fix 3: SCENE_DATASET Already Set in Task Config
**File**: [habitat_extensions/config/rxr_vlnce_english_task.yaml](habitat_extensions/config/rxr_vlnce_english_task.yaml#L8)

```yaml
SIMULATOR:
  TURN_ANGLE: 30
  TILT_ANGLE: 30
  ACTION_SPACE_CONFIG: v1
  SCENE_DATASET: "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"  # ✓ Already set
  AGENT_0:
    ...
```

## Verification Checklist

Before running training, verify:

- [ ] Scene dataset config file exists:
  ```bash
  ls -la data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
  ```

- [ ] Semantic files exist in at least one scene:
  ```bash
  ls -la data/scene_datasets/mp3d/1LXtFkjw3qL/ | grep -E "(\.house|_semantic\.ply|\.navmesh)"
  ```

- [ ] Your task CONFIG correctly references the dataset file:
  ```bash
  grep SCENE_DATASET habitat_extensions/config/*.yaml
  ```

## Expected Behavior (After Fix)

When you run training/evaluation:

1. ✅ Each episode loads its scene with proper semantic annotations
2. ✅ Different episodes get different semantic objects (if from different scenes)
3. ✅ Semantic sensor outputs unique object IDs per scene
4. ✅ Geodesic distances are computed correctly per episode
5. ✅ `norm_frames` will have different values based on object distances

## Debugging

To diagnose "same semantics" issue, run:

```bash
python /data/ws/VLN-CE/check_episodes.py          # Check if episodes are same/different scenes
python /data/ws/VLN-CE/debug_semantic_issue.py    # Check if semantics are properly loaded
python /data/ws/VLN-CE/debug_sensors.py           # Check sensor outputs
```

## Why "Same Semantics" Happened

The issue likely occurred because:

1. **Before the fix**: `scene_dataset_config_file` was never set on the simulator
2. **Result**: Semantic annotations weren't loaded, so `semantic_annotations()` returned empty or cached data
3. **Effect**: All episodes looked the same because semantic scene data wasn't being reloaded
4. **Now**: Each episode properly loads its scene's semantic configuration on reset/reconfigure

## Files Modified

1. ✅ [habitat-lab/habitat/config/default.py](habitat-lab/habitat/config/default.py) - Added SCENE_DATASET config
2. ✅ [habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py](habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py) - Pass SCENE_DATASET to simulator
3. ✅ [habitat_extensions/config/rxr_vlnce_english_task.yaml](habitat_extensions/config/rxr_vlnce_english_task.yaml) - Already has SCENE_DATASET

## Next Steps

1. Run the diagnostic scripts to confirm everything is working
2. Test that semantic observations differ between episodes
3. Verify geodesic distances are computed correctly
4. Run your training/evaluation pipeline
