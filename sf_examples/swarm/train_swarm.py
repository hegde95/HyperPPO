"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys
from typing import Any

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models, QuadMultiEncoder
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np
import torch
from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.context import global_model_factory


def add_more_quadrotors_env_args(env, parser):
    # in case we need to add more args in the future
    p = parser
    p.add_argument(
        "--env_agents",
        default=48,
        type=int,
        help="Num. agents in a vectorized env",
    )

# baseline model
class AppendedQuadMultiEncoder(QuadMultiEncoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg, obs_space)
        self.appended_fc = torch.nn.Linear(512, 4)

    def get_out_size(self) -> int:
        return 4
    
    def forward(self, obs_dict):
        out = super().forward(obs_dict)
        out = self.appended_fc(out)
        return out

def make_quadmulti_encoder(cfg, obs_space) -> Encoder:
    return AppendedQuadMultiEncoder(cfg, obs_space)


def register_models():
    global_model_factory().register_encoder_factory(make_quadmulti_encoder)


def override_params_for_batched_sampling(cfg):
    cfg.serial_mode = True
    cfg.async_rl = False
    cfg.batched_sampling = True
    cfg.num_workers = 1
    cfg.num_envs_per_worker = 1
    cfg.worker_num_splits = 1

    cfg.encoder_mlp_layers = [512, 512, 4]
    cfg.batch_size = 6144
    cfg.meta_batch_size = 16
    

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
    
class SimpleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0], info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step([action])
        return obs[0], reward[0], done[0], truncated[0], info[0]
    
    
def make_parallel_quadrotor_env(env_name, cfg, env_config, render_mode):
    def make_env(rank):
        def _init():
            env = make_quadrotor_env(env_name, cfg, _env_config = env_config, evaluation=False)
            env = SimpleWrapper(env)
            # env.seed(cfg.seed + rank)
            # env = TorchWrapper(env, cfg.env.num_agents)
            return env
        return _init
    return TorchWrapper(SubprocVecEnv([make_env(i) for i in range(cfg.env_agents)]), num_agents=cfg.env_agents)


def register_swarm_components(cfg):
    
    register_env("quadrotor_multi", make_parallel_quadrotor_env)
    if not cfg.hyper:
        register_models()

def parse_swarm_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    add_more_quadrotors_env_args(partial_cfg.env, parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    """Script entry point."""
    SINGLE_CLI = QUAD_BASELINE_CLI + (
        ' '
        # ' --async_rl=False --serial_mode=True --num_workers=16 --num_envs_per_worker=2 --rollout=128 --batch_size=2048 '
        # '--num_batches_per_epoch=4 '
    ) + (
        " ".join(sys.argv[1:])
    )
    cfg = parse_swarm_cfg(argv=SINGLE_CLI.split()[3:], evaluation=False)
    override_params_for_batched_sampling(cfg)
    register_swarm_components(cfg)
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())

    # python -m sf_examples.swarm.train_swarm --env quadrotor_multi --experiment test3 --train_dir dummy --train_for_env_steps 100_000_000 --dual_critic False --multi_stddev True --arch_sampling_mode biased --hyper False --with_wandb True --wandb_tags debug