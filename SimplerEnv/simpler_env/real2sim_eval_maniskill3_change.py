from collections import defaultdict
import json
import os
import signal
import time
import numpy as np
from typing import Annotated, Optional

import torch
import tree
from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
import tyro
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf
from simpler_env.text_processing import text_processors
from transformers import FlaxCLIPModel

def tensor_to_numpy_info(np_data) -> dict:
    # print(type(np_data))
    # print(np_data)
    # 移至 CPU 并转为 NumPy
    if isinstance(np_data, torch.Tensor):
        np_data = np_data.cpu().numpy()
    elif isinstance(np_data, tf.Tensor):
        np_data = np_data.numpy()
    else:
        np_data = np_data
    if isinstance(np_data, bytes):
        return {
            "data": np_data.decode('utf-8'),  
            "shape": (),
            "dtype": 'string'  
        }
    else:
        return {
            "data": np_data.tolist(),  
            "shape": np_data.shape,
            "dtype": np_data.dtype.name  # 如 'int32', 'float32'
        }

@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments. Example command to run: 

    XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
        --model="octo-small" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 192 --num-envs 64
    """


    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    shader: str = "default"

    num_envs: int = 1
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    video_dir: str = "videos"
    """The directory to save videos and results"""

    model: Optional[str] = None
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models. Only used for RT models"""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    reset_by_episode_id: bool = True
    """Whether to reset by fixed episode ids instead of random sampling initial states."""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    debug: bool = False

    record_dir: str = "record"
    """The directory to save results"""

def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.record_dir, exist_ok=True) 
    sensor_configs = dict()
    sensor_configs["shader_pack"] = args.shader
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs
    )
    sim_backend = 'gpu' if env.device.type == 'cuda' else 'cpu'

    # Setup up the policy inference model   
    model = None
    try:

        policy_setup = "widowx_bridge"
        if args.model is None:
            pass
        else:
            from simpler_env.policies.rt1.rt1_model import RT1Inference
            from simpler_env.policies.octo.octo_model import OctoInference
            if args.model == "octo-base" or args.model == "octo-small":
                model = OctoInference(model_type=args.model, policy_setup=policy_setup, init_rng=args.seed, action_scale=1)
            elif args.model == "rt-1x":
                ckpt_path=args.ckpt_path
                model = RT1Inference(
                    saved_model_path=ckpt_path,
                    policy_setup=policy_setup,
                    action_scale=1,
                )
            elif args.model is not None:
                raise ValueError(f"Model {args.model} does not exist / is not supported.")
    except:
        if args.model is not None:
            raise Exception("SIMPLER Env Policy Inference is not installed")

    model_name = args.model if args.model is not None else "random"
    if model_name == "random":
        print("Using random actions.")
    exp_dir = os.path.join(args.video_dir, f"real2sim_eval/{model_name}_{args.env_id}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)
    eps_count = 0

    print(f"Running Real2Sim Evaluation of model {args.model} on environment {args.env_id}")
    print(f"Using {args.num_envs} environments on the {sim_backend} simulation backend")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()
    
    while eps_count < args.num_episodes:
        # os.makedirs(os.path.join(args.record_dir, f"episode_{eps_count}"), exist_ok=True)
        seed = args.seed + eps_count
        print(f"seed: {seed}")
        obs, _ = env.reset(seed=seed, options={"episode_id": torch.tensor([seed + i for i in range(args.num_envs)])})
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction[0])
        processor = text_processors["clip_processor"]()
        inputs = processor.encode(["a photo of a cat"])
        text_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        text_embeds = text_model.get_text_features(**inputs)
        if model is not None:
            model.reset(instruction)
        images = []
        predicted_terminated, truncated = False, False
        images.append(get_image_from_maniskill3_obs_dict(env, obs))
        elapsed_steps = 0
        while not (predicted_terminated or truncated):
            if model is not None:
                start_time = time.time()
                raw_action, action = model.step(images[-1], instruction)
                base_action = action
                action = torch.cat([action["world_vector"], action["rot_axangle"], action["gripper"]], dim=1)
                timers["inference"] += time.time() - start_time
            else:
                action = env.action_space.sample()

            if elapsed_steps > 0:
                if args.save_video and args.info_on_video:
                    for i in range(len(images[-1])):
                        images[-1][i] = visualization.put_info_on_image(images[-1][i], tree.map_structure(lambda x: x[i], info))
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            for env_idx in range(args.num_envs):
                    episode_dir = os.path.join(args.record_dir, f"episode_{seed + env_idx}")
                    os.makedirs(episode_dir, exist_ok=True)
                    log_filename = os.path.join(episode_dir, f"step_{elapsed_steps}.json")
                    with open(log_filename, "a") as f:
                        terminate_val = base_action.get('terminate_episode', np.array([0.]))
                        is_terminated = bool(terminate_val.item() == 1)
                        env_action = {
                            'open_gripper': tensor_to_numpy_info(raw_action['open_gripper'][env_idx]),
                            'rotation_delta': tensor_to_numpy_info(raw_action['rotation_delta'][env_idx]),
                            'terminate_episode': tensor_to_numpy_info(base_action['terminate_episode'][env_idx]),
                            'world_vector': tensor_to_numpy_info(raw_action['world_vector'][env_idx])
                            
                        }
                        # print(env_action["open_gripper"].)
                        f.write(f"('action', {env_action})\n")
                        f.write(f"('is_first', {(tensor_to_numpy_info(tf.constant(bool((elapsed_steps == 0)))))})\n")
                        f.write(f"('is_last', {(tensor_to_numpy_info(tf.constant(bool((elapsed_steps == 119)))))})\n")
                        f.write(f"('is_terminated', {tensor_to_numpy_info(tf.constant(is_terminated))})\n")
                        # print(f"obs: {obs}")
                        # input()
                        env_obs = {
                            'image': tensor_to_numpy_info(images[-1][env_idx]),
                            'natural_language_embedding': tensor_to_numpy_info(text_embeds),
                            'natural_language_instruction': tensor_to_numpy_info(tf.constant(instruction[0], dtype=tf.string)),
                            'state': tensor_to_numpy_info(obs['agent']['controller']['arm']['target_pose'][env_idx])
                        }
                        f.write(f"('obs', {env_obs})\n")
                        f.write(f"('reward', {tensor_to_numpy_info(reward[env_idx])})\n")
            timers["env.step"] += time.time() - start_time
            elapsed_steps += 1
            info = common.to_numpy(info)
            
            truncated = bool(truncated.any()) # note that all envs truncate and terminate at the same time.
            images.append(get_image_from_maniskill3_obs_dict(env, obs))

        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
        if args.save_video:
            for i in range(len(images[-1])):
                images_to_video([img[i].cpu().numpy() for img in images], exp_dir, f"{sim_backend}_eval_{seed + i}_success={info['success'][i].item()}", fps=10, verbose=True)
        eps_count += args.num_envs
        if args.num_envs == 1:
            print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        else:
            print(f"Evaluated {args.num_envs} episodes, seeds {seed} to {eps_count}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")
    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]
    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["total_episodes"] = eps_count
    mean_metrics["time/episodes_per_second"] = eps_count / timers["total"]
    print("Timing Info:")
    for key, value in timers.items():
        mean_metrics[f"time/{key}"] = value
        print(f"{key}: {value:.2f} seconds")
    metrics_path = os.path.join(exp_dir, f"{sim_backend}_eval_metrics.json")
    if sim_backend == "gpu":
        metrics_path = metrics_path.replace("gpu", f"gpu_{args.num_envs}_envs")
    with open(metrics_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(f"Evaluation complete. Results saved to {exp_dir}. Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()