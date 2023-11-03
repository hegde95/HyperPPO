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



class RandomizedMTEnv:
    def __init__(self,eval=False, render_mode=None):
        
        if not eval and render_mode == "human":
            raise ValueError("Cannot render metaworld envs in training mode.")

        self.eval = eval
        self.render_mode = render_mode

        self.env_dict = {}
        self.env_names = list(MT10_ENV_NAMES_MAP.keys())
        self.num_envs = len(self.env_names)

        if not self.eval:
            for env_name in self.env_names:
                self.env_dict[env_name] = get_env(env_name)

            self.action_space = self.env_dict[self.env_names[0]].action_space
            self.observation_space = self.env_dict[self.env_names[0]].observation_space
        
        else:
            dummy_env = get_env(self.env_names[0], render_mode=None)

            self.action_space = dummy_env.action_space
            self.observation_space = dummy_env.observation_space


    def reset(self,):
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
        obs_dict = {}
        obs_dict['obs'] = self.env.reset()[0]
        obs_dict['task'] = self.current_task
        info = {}
        return obs_dict, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs_dict = {}
        obs_dict['obs'] = obs
        obs_dict['task'] = self.current_task
        return obs, reward, done, truncated, info
    
    def render(self, mode="human"):
        return self.env.render()
    

def make_parallel_metaworld_env(env_name, cfg, env_config, render_mode):
    def make_env(rank):
        def _init():
            env = RandomizedMTEnv()
            return env
        return _init
    return SubprocVecEnv([make_env(i) for i in range(cfg.env_agents)])

        

def register_metaworld_components(cfg):
    
    register_env("metaworld_multi", make_parallel_metaworld_env)


def parse_metaworld_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
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
