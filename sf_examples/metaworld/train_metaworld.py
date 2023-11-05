import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from time import sleep
import copy
import torch
from typing import Any
import gymnasium as gym
from sf_examples.metaworld.metaworld_params import add_metaworld_env_args, metaworld_override_defaults
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder
import torch.nn as nn
from typing import List
from sample_factory.algo.utils.torch_utils import calc_num_elements

MT10_ENV_NAMES_MAP = {
    'reach':0,
    'push':1,
    'pick-place':2,
    'door-open':3,
    'drawer-open':4,
    'drawer-close':5,
    'button-press-topdown':6,
    'peg-insert-side':7,
    'window-open':8,
    'window-close':9,
}


def get_env(env_name: str, render_mode=None):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=0, render_mode=render_mode)
    env.seeded_rand_vec = True
    env = TimeLimit(env, max_episode_steps=500)
    return env

class TorchWrapper(gym.Wrapper):

    def __init__(self, env, num_agents, is_obs_dict=False):
        self.num_agents = num_agents
        super().__init__(env)
        self.is_obs_dict = is_obs_dict

    def step(self, action):
        # action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        truncated = done
        reward = torch.tensor(reward).float()
        done = torch.tensor(done)
        truncated = torch.tensor(truncated)
        return self.make_tensor_dict(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.make_tensor_dict(obs), torch.zeros(1)
    

    def make_tensor_dict(self, obs):
        if self.is_obs_dict:
            obs_dict = {}
            # obs_dict['obs'] = obs[:,:-1]
            # obs_dict['task'] = obs[:,-1].unsqueeze(1).to(torch.long)
            for k,v in obs.items():
                obs_dict[k] = torch.from_numpy(v)

                if k == 'task':
                    obs_dict[k] = obs_dict[k].to(torch.long)
            return obs_dict
        else:
            return torch.from_numpy(obs)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed=seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

class MlpEncoder(Encoder):
    def __init__(self, cfg, inp_dim):
        super().__init__(cfg)

        self.mlp_head = nn.Sequential(
            nn.Linear(inp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.encoder_out_size = 256

    def forward(self, obs):
        x = self.mlp_head(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size
    
class MTEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0
        self.encoders['task'] = nn.Sequential(
            nn.Embedding(10, 10),
            MlpEncoder(cfg, 10)
        )
        out_size += self.encoders['task'][1].get_out_size()

        if 'state' in self.obs_keys:
            self.encoders['state'] = MlpEncoder(cfg, obs_space['state'].shape[0])
        else:
            self.encoders['state'] = MlpEncoder(cfg, obs_space['obs'].shape[0]-1)
        out_size += self.encoders['state'].get_out_size()

        self.fc_out = nn.Sequential(
            nn.Linear(out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

        self.encoder_out_size = 4

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            if key == 'task':
                x = self.encoders[key](obs_dict[key].to(torch.long)).squeeze(1)
            else:    
                x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        enc = torch.cat(encodings, 1)
        out = self.fc_out(enc)
        return out

    def get_out_size(self) -> int:
        return self.encoder_out_size

def make_mt_encoder(cfg, obs_space) -> Encoder:
    return MTEncoder(cfg, obs_space)

def register_models():
    global_model_factory().register_encoder_factory(make_mt_encoder)

class RandomizedMTEnv(gym.Env):
    def __init__(self,eval=False, render_mode=None, is_obs_dict=False):
        
        if not eval and render_mode == "human":
            raise ValueError("Cannot render metaworld envs in training mode.")

        self.eval = eval
        self.render_mode = render_mode
        self.is_obs_dict = is_obs_dict

        self.env_dict = {}
        self.env_names = list(MT10_ENV_NAMES_MAP.keys())
        self.num_envs = len(self.env_names)

        if not self.eval:
            for env_name in self.env_names:
                self.env_dict[env_name] = get_env(env_name)

            self.action_space = self.env_dict[self.env_names[0]].action_space
            single_task_obs_space = self.env_dict[self.env_names[0]].observation_space

            self.observation_space = self.get_observation_space(single_task_obs_space)
        
        else:
            dummy_env = get_env(self.env_names[0], render_mode=None)

            self.action_space = dummy_env.action_space
            single_task_obs_space = dummy_env.observation_space
            self.observation_space = self.get_observation_space(single_task_obs_space)

    def get_observation_space(self, single_task_obs_space):
        if self.is_obs_dict:
            task_space = gym.spaces.Box(low=0, high=self.num_envs-1, shape=(1,), dtype=np.int64)
            combined_space = gym.spaces.Dict({'state':single_task_obs_space, 'task':task_space})
            return combined_space
        else:
            state_space = single_task_obs_space
            task_space = gym.spaces.Box(low=0, high=self.num_envs-1, shape=(1,), dtype=np.int64)
            
            combined_space_low = np.concatenate([state_space.low, task_space.low])
            combined_space_high = np.concatenate([state_space.high, task_space.high])

            combined_space = gym.spaces.Box(low=combined_space_low, high=combined_space_high, dtype=np.float32)

            return combined_space


    def make_observation(self, obs, task):
        if self.is_obs_dict:
            obs_dict = {}
            obs_dict['state'] = obs
            obs_dict['task'] = task
            return obs_dict
        else:
            cat_obs = np.concatenate([obs, task])  
            return cat_obs          


    def reset(self, **kwargs):
        # select a random env
        self.env_name = np.random.choice(self.env_names)
        self.current_task = np.array([MT10_ENV_NAMES_MAP[self.env_name]])

        if not self.eval:
            self.env = self.env_dict[self.env_name]
        else:
            # close the previous env
            if hasattr(self, 'env'):
                self.env.close()
                del self.env
                self.env = get_env(self.env_name, render_mode=self.render_mode)


            else:
                self.env = get_env(self.env_name, render_mode=self.render_mode)


        # reset the env
        obs = self.env.reset(**kwargs)[0]
        final_obs = self.make_observation(obs, self.current_task)

        info = {}
        return final_obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        final_obs = self.make_observation(obs, self.current_task)
        return final_obs, reward, done, truncated, info
    
    def render(self, mode="human"):
        return self.env.render()
    

def make_parallel_metaworld_env(env_name, cfg, env_config, render_mode):
    is_obs_dict = True
    def make_env(rank, is_obs_dict):
        def _init():
            env = RandomizedMTEnv(is_obs_dict = is_obs_dict)
            return env
        return _init
    return TorchWrapper(SubprocVecEnv([make_env(i, is_obs_dict = True) for i in range(cfg.env_agents)]), num_agents = cfg.env_agents, is_obs_dict = True)

        

def register_metaworld_components():
    
    register_env("metaworld_multi", make_parallel_metaworld_env)
    register_models()


def parse_metaworld_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_metaworld_env_args(partial_cfg.env, parser)
    metaworld_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_metaworld_components()
    cfg = parse_metaworld_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
# python -m sf_examples.metaworld.train_metaworld --env metaworld_multi --experiment test --restart_behavior overwrite --train_dir dummy --train_for_env_steps 4_000_000 --with_wandb True --wandb_tags debug2 --dual_critic False --multi_stddev True --arch_sampling_mode biased --hyper False --env_agents 4