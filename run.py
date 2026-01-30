#!/usr/bin/env python3

import argparse
import os
import random
from icecream import ic
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from vlnce_baselines.nonlearning_agents import (
    evaluate_agent,
    nonlearning_inference,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference", "sim_output"],
        required=True,
        help="run type of the experiment (train, eval, inference, sim_output)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def get_simulator_output(config) -> None:
    """Extract and display simulator output.
    
    Args:
        config: Configuration object
    """
    from habitat import Env
    from habitat_baselines.common.environments import get_env_class
    from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
    from vlnce_baselines.common.utils import extract_instruction_tokens
    from habitat_baselines.common.obs_transformers import (
        apply_obs_transforms_batch,
        get_active_obs_transforms,
    )
    from habitat_baselines.utils.common import batch_obs
    
    logger.info("Getting simulator output...")
    import imageio
    # config.defrost()
    # config.NUM_PROCESSES = 1
    # config.freeze()
    # Create environment
    # envs = construct_envs_auto_reset_false(
    #     config, get_env_class(config.ENV_NAME)
    # )

    
    env= Env(config=config.TASK_CONFIG)
    # Defrost config to disable text features
    
    # # Get observation transforms
    # obs_transforms = get_active_obs_transforms(config)
    
    # # Reset environment and get initial observations
    # observations = envs.reset()
    # observations = extract_instruction_tokens(
    #     [observations], config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
    # )
    # observations = observations[0]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # logger.info("=" * 50)
    # logger.info("STARTING EPISODE")
    # logger.info("=" * 50)
    
    # step_count = 0
    # done = False
    
    # while not done:
    #     # Process observation
    #     batch = batch_obs([observations], device)
    #     batch = apply_obs_transforms_batch(batch, obs_transforms)
        
    #     logger.info(f"\n--- Step {step_count} ---")
    #     logger.info("SIMULATOR OUTPUT")
    #     for key, value in batch.items():
    #         if isinstance(value, torch.Tensor):
    #             logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
    #         else:
    #             logger.info(f"{key}: {type(value)}")
        
    #     # Take a random action
    #     action = envs.action_space.sample()
    #     observations, reward, done = envs.step(action)
        
    #     logger.info(f"Action: {action}, Reward: {reward}, Done: {done}")
        
    #     step_count += 1
    
    # logger.info("=" * 50)
    # logger.info(f"EPISODE COMPLETE - Total steps: {step_count}")
    # logger.info("=" * 50)
    
    # envs.close()
    # logger.info("Simulator output extraction complete.")
    # observations = env.reset()
    # frames = []
    # # from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
    # # trainer = BaseVLNCETrainer(config)
    # # obs, act= trainer._get_spaces(config, env)

    # done = False
    # i= 0
    # while not done:
    #     # obs = env.get_observations()
    #     observations = extract_instruction_tokens(
    #         [observations], config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
    #     )
    #     observations = observations[0]
    #     print(">>>>>>", observations.keys())

    #     rgb = observations["rgb"]
    #     frames.append(rgb)

    #     # # get GT oracle 
    #     #NOTE: eed to implement this
    #     action = env.task.get_oracle_action()

    #     observations = env.step(action)
    #     done = env.episode_over
    #     i+=1
    #     if i>100:
    #         break

    # imageio.mimsave("gt_episode.mp4", frames, fps=10)


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train", "eval", "inference", or "sim_output".
        opts: list of strings of additional config options.
    """
    config = get_config(exp_config, opts)
    logger.info(f"config: {config}")
    logdir = "/".join(config.LOG_FILE.split("/")[:-1])
    if logdir:
        os.makedirs(logdir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)


    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    
    if run_type == "sim_output":
        get_simulator_output(config)
        return
    
    if run_type == "eval":
        torch.backends.cudnn.deterministic = True
        if config.EVAL.EVAL_NONLEARNING:
            evaluate_agent(config)
            return

    if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
        nonlearning_inference(config)
        return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()


if __name__ == "__main__":
    main()