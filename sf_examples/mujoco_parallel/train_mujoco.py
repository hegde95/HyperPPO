import sys
from typing import Any

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.mujoco_parallel.mujoco_params import add_mujoco_env_args, mujoco_override_defaults
from sf_examples.mujoco_parallel.mujoco_utils import MUJOCO_ENVS, make_mujoco_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np
import torch



# this class converts all state and action from numpy to torch
class TorchWrapper(gym.Wrapper):
    def __init__(self, env, num_agents):
        super().__init__(env)
        self.num_agents = num_agents
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.observation_space.shape, dtype=np.float32)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)

    def step(self, action):
        # action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        truncated = done
        obs = torch.from_numpy(obs).float()
        reward = torch.tensor(reward).float()
        done = torch.tensor(done)
        truncated = torch.tensor(truncated)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float()
        # obs_dict = {}
        # obs_dict['obs'] = obs
        # obs_dict['obs']['obs'] = obs
        info = torch.zeros(1)
        return obs, info
    
    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def make_parallel_mujoco_env(env_name, cfg, env_config, render_mode):
    def make_env(rank):
        def _init():
            env = make_mujoco_env(env_name, cfg, _env_config = env_config, evaluation=False)
            # env.seed(cfg.seed + rank)
            return env
        return _init
    return TorchWrapper(SubprocVecEnv([make_env(i) for i in range(cfg.env_agents)]), num_agents=cfg.env_agents)

def register_mujoco_components():
    for env in MUJOCO_ENVS:
        register_env(env.name, make_parallel_mujoco_env)


def parse_mujoco_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_mujoco_env_args(partial_cfg.env, parser)
    mujoco_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
