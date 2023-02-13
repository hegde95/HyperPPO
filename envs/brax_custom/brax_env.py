import functools
import gym
import brax

from envs import brax_custom
from brax.envs import to_torch
from jax.dlpack import to_dlpack
from envs.wrappers.normalize_torch import NormalizeObservation, NormalizeReward

import torch
v = torch.ones(1, device='cuda:0')  # init torch cuda before jax

_to_custom_env = {
    'ant': {'custom_env_name': 'brax_custom-ant-v0',
            'action_clip': (-1, 1),
            'reward_clip': (-10, 10),
            'obs_clip': (-10, 10)},
    'humanoid': {'custom_env_name': 'brax_custom-humanoid-v0',
                 'action_clip': (-1, 1),
                 'reward_clip': (-10, 10),
                 'obs_clip': (-10, 10)},
    'walker2d': {'custom_env_name': 'brax-custom-walker2d-v0',
                 'action_clip': (-1, 1),
                 'reward_clip': (-10, 10),
                 'obs_clip': (-10, 10)},
    'halfcheetah': {'custom_env_name': 'brax-custom-halfcheetah-v0',
                    'action_clip': (-1, 1)}
}


def make_vec_env_brax(env_name, env_batch_size, seed, device):
    entry_point = functools.partial(brax_custom.create_gym_env, env_name=env_name)
    brax_env_name = _to_custom_env[env_name]['custom_env_name']
    if brax_env_name not in gym.envs.registry.env_specs:
        gym.register(brax_env_name, entry_point=entry_point)

    act_bounds = _to_custom_env[env_name]['action_clip']
    vec_env = gym.make(_to_custom_env[env_name]['custom_env_name'], batch_size=env_batch_size, seed=seed,
                       clip_actions=act_bounds)
    vec_env = to_torch.JaxToTorchWrapper(vec_env, device=device)
    vec_env = NormalizeObservation(vec_env)
    vec_env = NormalizeReward(vec_env)

    return vec_env
