import json
import os
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from icecream import ic

import jsonlines
import torch
import torch.nn.functional as F
import tqdm
import cv2
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs

from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401


class BaseVLNCETrainer(BaseILTrainer):
    """A base trainer for VLN-CE imitation learning."""

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.IL.lr
        )
        if load_from_ckpt:
            ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu", weights_only= False)
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.start_epoch = ckpt_dict["epoch"] + 1
                self.step_id = ckpt_dict["step_id"]
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    def _get_spaces(
        self, config: Config, envs: Optional[Any] = None
    ) -> Tuple[Space]:
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(self.config.ENV_NAME)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space

    def save_checkpoint(self, file_name: str) -> None:
        """Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint
        """
        checkpoint = {
            "state_dict": self.policy.state_dict(),
            "config": self.config,
        }
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        T, N = corrected_actions.size()

        recurrent_hidden_states = torch.zeros(
            N,
            self.policy.net.num_recurrent_layers,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        distribution = self.policy.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss = loss / loss_accumulation_scalar
        loss.backward()

        if step_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        from habitat import Env
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)
        # envs = Env(config=config.TASK_CONFIG)
    
        # observation_space = envs.observation_space
        # action_space = envs.action_space


        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()
        #NOTE: Load Gt trajectories
        gt_trajectories = self._load_gt_trajectories(
        config.EVAL.SPLIT
        )
        step_counters = [0 for _ in range(envs.num_envs)]
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()
            current_episode= current_episodes[0]
            trajectory_id= current_episode.trajectory_id
            with torch.no_grad():
                policy_actions, rnn_states = self.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=not config.EVAL.SAMPLE,
                )
                gt_actions = torch.zeros_like(policy_actions)
        
                for i in range(envs.num_envs):
                    ep_id = current_episodes[i].trajectory_id
                    
                    if ep_id in gt_trajectories:
                        gt_actions_list = gt_trajectories[ep_id]  # List: [1, 1, 2, 1, 0]
                        current_step = step_counters[i]
                        
                        if current_step < len(gt_actions_list):
                            # Use GT action for this step
                            gt_action = gt_actions_list[current_step]
                            gt_actions[i][0] = gt_action
                            
                            logger.info(
                                f"Env {i}, Step {current_step}: "
                                f"Policy={policy_actions[i][0].item()}, "
                                f"GT={gt_action}"
                            )
                        else:
                            # Out of GT actions, use STOP
                            gt_actions[i][0] = 0
                            logger.warning(f"Env {i}: Out of GT actions, using STOP")
                        
                        step_counters[i] += 1
                    else:
                        # No GT available, fallback to policy
                        gt_actions[i] = policy_actions[i]
                        logger.warning(f"Env {i}: No GT for {ep_id}, using policy")

            prev_actions.copy_(gt_actions)

            outputs = envs.step([a[0].item() for a in gt_actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(envs.num_envs):                
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)
                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)
                step_counters[i] = 0
                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)
    
    #NOTE load GT_trajectories
    def _load_gt_trajectories(self, split: str = "val_unseen") -> Dict:
        """Load GT trajectories from file.
        
        Args:
            split: data split name (e.g., "val_unseen")
        
        Returns:
            Dict mapping episode_id to trajectory data
        """
        import gzip
        import json
        
        gt_file = os.path.join(
            self.config.GT_FOLDER,
            split,
            f"{split}_guide_gt.json.gz"
        )
        
        if not os.path.exists(gt_file):
            logger.warning(f"GT trajectories file not found: {gt_file}")
            return {}
        
        logger.info(f"Loading GT trajectories from: {gt_file}")
        
        try:
            with gzip.open(gt_file, 'rt', encoding='utf-8') as f:
                full_json= json.load(f)
                return {
                key: value.get("actions", []) 
                for key, value in full_json.items()
            }
            
            
        except Exception as e:
            logger.error(f"Error loading GT trajectories: {e}")
            return {}
        
        # return gt_trajectories
    #NOTE: shortest path waypoint_to_action
    def _waypoints_to_actions(
        self,
        waypoints: List[List[float]],
        current_position: List[float],
        current_rotation: List[float],
    ) -> int:
        """Convert current position and next waypoint to action.
        
        Args:
            waypoints: List of [x, y, z] positions (GT path)
            current_position: Current agent [x, y, z]
            current_rotation: Current agent rotation (quaternion)
        
        Returns:
            action: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        """
        
        import numpy as np
        
        # Get next waypoint
        if len(waypoints) == 0:
            return 0  # STOP
        
        next_waypoint = np.array(waypoints[0])
        current_pos = np.array(current_position)
        
        # Calculate distance to next waypoint
        distance = np.linalg.norm(next_waypoint[:2] - current_pos[:2])  # Only XZ distance
        
        logger.info(f"Distance to next waypoint: {distance:.3f}m")
        
        # If reached waypoint (within 0.3m), remove it and go to next
        if distance < 0.3:
            waypoints.pop(0)  # Remove reached waypoint
            if len(waypoints) == 0:
                return 0  # STOP - reached end
            next_waypoint = np.array(waypoints[0])
            distance = np.linalg.norm(next_waypoint[:2] - current_pos[:2])
        
        # Get agent's forward direction from rotation (quaternion)
        # quaternion: [x, y, z, w]
        qx, qy, qz, qw = current_rotation
        
        # Convert quaternion to forward vector
        forward = np.array([
            2 * (qw * qy + qx * qz),  # x component
            2 * (qz * qy - qx * qw),  # z component
        ])
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        # Direction to next waypoint
        direction_to_waypoint = next_waypoint[:2] - current_pos[:2]
        direction_to_waypoint = direction_to_waypoint / (np.linalg.norm(direction_to_waypoint) + 1e-6)
        
        # Calculate angle between forward and target direction
        # Using cross product and dot product
        dot_product = np.dot(forward, direction_to_waypoint)
        cross_product = forward[0] * direction_to_waypoint[1] - forward[1] * direction_to_waypoint[0]
        
        angle = np.arctan2(cross_product, dot_product)  # -π to π
        angle_degrees = np.degrees(angle)
        
        logger.info(f"Angle to waypoint: {angle_degrees:.1f}°")
        
        # Decide action based on angle
        # Agent turns 15 degrees per LEFT or RIGHT action
        
        if abs(angle_degrees) < 15:  # Already facing waypoint
            return 1  # FORWARD
        elif angle_degrees > 0:  # Need to turn left
            return 2  # LEFT
        else:  # Need to turn right
            return 3  # RIGHT
        
    #NOTE: This is original inference implementation
    def inference(self) -> None:
        """Runs inference on a checkpoint and saves a predictions file."""
        import os
        from habitat_extensions.utils import generate_video, observations_to_image
        from habitat.utils.visualizations.utils import append_text_to_image
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")[
                    "config"
                ]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.INFERENCE.LANGUAGES
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = config.INFERENCE.CKPT_PATH
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = [
            s for s in config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s
        ]
        config.ENV_NAME = "VLNCEInferenceEnv"

        # Add TOP_DOWN_MAP for visualization
        config.TASK_CONFIG.TASK.MEASUREMENTS = ["TOP_DOWN_MAP_VLNCE"]
        config.ENV_NAME = "VLNCEInferenceEnv"
        
        # # Video settings
        config.VIDEO_DIR = os.path.join(config.CHECKPOINT_FOLDER, "inference_videos")
        config.VIDEO_OPTION = ["disk"]
        
        config.freeze()

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )

        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        episode_predictions = defaultdict(list)

        # episode ID --> instruction ID for rxr predictions format
        instruction_ids: Dict[str, int] = {}
        # Create video directory
        os.makedirs(config.VIDEO_DIR, exist_ok=True)

        # Store RGB frames and metadata for each environment
        rgb_frames = [[] for _ in range(envs.num_envs)]
        episode_instructions = {}
        episode_data = {}

        # populate episode_predictions with the starting state
        current_episodes = envs.current_episodes()
        for i in range(envs.num_envs):
            episode = current_episodes[i]
            episode_instructions[episode.episode_id] = {
                "text": episode.instruction.instruction_text,
                "tokens": episode.instruction.instruction_tokens,
            }
        for i in range(envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                envs.call_at(i, "get_info", {"observations": {}})
            )
            if config.INFERENCE.FORMAT == "rxr":
                ep_id = current_episodes[i].episode_id
                k = current_episodes[i].instruction.instruction_id
                instruction_ids[ep_id] = int(k)

        with tqdm.tqdm(
            total=sum(envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while envs.num_envs > 0:
                current_episodes = envs.current_episodes()
                with torch.no_grad():
                    actions, rnn_states = self.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=not config.INFERENCE.SAMPLE,
                    )
                    prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(envs.num_envs):
                    episode_predictions[current_episodes[i].episode_id].append(
                        infos[i]
                    )
                    if not dones[i]:
                        continue

                    observations[i] = envs.reset_at(i)[0]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update()

                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                envs_to_pause = []
                next_episodes = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue

                    if next_episodes[i].episode_id in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[
                            next_episodes[i].episode_id
                        ].append(
                            envs.call_at(i, "get_info", {"observations": {}})
                        )
                        if config.INFERENCE.FORMAT == "rxr":
                            ep_id = next_episodes[i].episode_id
                            k = next_episodes[i].instruction.instruction_id
                            instruction_ids[ep_id] = int(k)

                (
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                    _,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                )

        envs.close()

        if config.INFERENCE.FORMAT == "r2r":
            with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(episode_predictions, f, indent=2)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
        else:  # use 'rxr' format for rxr-habitat leaderboard
            predictions_out = []

            for k, v in episode_predictions.items():

                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if path[-1] != p["position"]:
                        path.append(p["position"])

                predictions_out.append(
                    {
                        "instruction_id": instruction_ids[k],
                        "path": path,
                    }
                )

            predictions_out.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(
                config.INFERENCE.PREDICTIONS_FILE, mode="w"
            ) as writer:
                writer.write_all(predictions_out)

            logger.info(
                f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}"
            )
    