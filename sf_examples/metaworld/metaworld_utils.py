from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from gym.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch
from typing import Any
import gymnasium as gym
from collections import deque
from sample_factory.utils.utils import log




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


def get_env(env_name: str, render_mode=None, seed=0):
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-v2-goal-observable"](seed=seed, render_mode=render_mode)
    env.seeded_rand_vec = True
    env = TimeLimit(env, max_episode_steps=500)
    return env



class RandomizedMTEnv(gym.Env):
    def __init__(self,eval=False, render_mode=None, is_obs_dict=False, env_name=None, seed=0):
        
        if not eval and render_mode == "human":
            raise ValueError("Cannot render metaworld envs in training mode.")
        
        self.eval = eval
        self.render_mode = render_mode
        self.is_obs_dict = is_obs_dict
        self.env_name = env_name

        self.set_seed(seed)

        self.ramdomize_env = True
        if self.env_name is not None:
            self.ramdomize_env = False
            log.info(f"Using the env {self.env_name} for training")
        else:
            log.info(f"Randomizing the env for training")

        self.env_names = list(MT10_ENV_NAMES_MAP.keys())
        self.num_envs = len(self.env_names)

        self.envs_dict_loaded = False
        if not self.eval and self.ramdomize_env:
            self.env_dict = {}
            
            # if its in training mode, and we need to randomize the env, then we need to create all the envs in a dict
            for env_name in self.env_names:
                self.env_dict[env_name] = get_env(env_name, render_mode=None, seed=self.rng.randint(0,1000))
            self.envs_dict_loaded = True

        # get the observation and action spaces
        self.observation_space, self.action_space = self.get_spaces(is_loaded_envs_dict=self.envs_dict_loaded)
        
    def set_seed(self, seed):
        # create a rng for each env
        self.rng = np.random.RandomState(seed)

    def get_spaces(self, is_loaded_envs_dict):
        if is_loaded_envs_dict:
            self.action_space = self.env_dict[self.env_names[0]].action_space
            single_task_obs_space = self.env_dict[self.env_names[0]].observation_space

            self.observation_space = self.get_observation_space(single_task_obs_space)
        else:
            dummy_env = get_env(self.env_names[0], render_mode=None)

            self.action_space = dummy_env.action_space
            single_task_obs_space = dummy_env.observation_space
            self.observation_space = self.get_observation_space(single_task_obs_space)
        
        return self.observation_space, self.action_space

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
        if self.ramdomize_env:
            # select a random env 
            self.env_name = np.random.choice(self.env_names)
        self.current_task = np.array([MT10_ENV_NAMES_MAP[self.env_name]])

        if self.envs_dict_loaded:
            self.env = self.env_dict[self.env_name]
        else:
            # close the previous env
            if hasattr(self, 'env'):
                self.env.close()
                del self.env

            self.env = get_env(self.env_name, render_mode=self.render_mode, seed=self.rng.randint(0,1000))


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
    


class TorchWrapper(gym.Wrapper):

    def __init__(self, env, num_agents, is_obs_dict=False):
        self.num_agents = num_agents
        super().__init__(env)
        self.is_obs_dict = is_obs_dict
        self.initialize_episode_stats()

    def step(self, action):
        if action.ndim == 1:
            action = action.reshape((1,-1))
        obs, reward, done, info = self.env.step(action)

        # convert info if it is a tuple
        if isinstance(info, tuple) and not (done.any()):
            info = info[0]

        truncated = done
        reward = torch.tensor(reward).float()
        done = torch.tensor(done)
        truncated = torch.tensor(truncated)

        if done.any() | truncated.any():
            info = self.log_episode_end_stats(info)
            
        return self.make_tensor_dict(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.make_tensor_dict(obs), torch.zeros(1)
    
    def log_episode_end_stats(self, info):
        infos = {
            k : np.array([info[i][k] for i in range(self.num_agents)])
            for k in info[0].keys()
        }         
        current_env_name = self.env.get_attr('env_name')
        for i, env_name in enumerate(current_env_name):
            self.task_wise_success[env_name].append(infos['success'][i])
            self.task_wise_near_object[env_name].append(infos['near_object'][i])
            self.task_wise_grasp_reward[env_name].append(infos['grasp_reward'][i])
            self.task_wise_unscaled_reward[env_name].append(infos['unscaled_reward'][i])
            self.task_wise_TimeLimitTruncated[env_name].append(infos['TimeLimit.truncated'][i])
            self.task_wise_grasp_success[env_name].append(infos['grasp_success'][i])
            self.task_wise_obj_to_target[env_name].append(infos['obj_to_target'][i])
            self.task_wise_in_place_reward[env_name].append(infos['in_place_reward'][i])
        
        results = {}
        for env_name in self.task_wise_success.keys():
            results[env_name + '_success'] = np.mean(self.task_wise_success[env_name]) if len(self.task_wise_success[env_name]) > 0 else 0
            results[env_name + '_near_object'] = np.mean(self.task_wise_near_object[env_name]) if len(self.task_wise_near_object[env_name]) > 0 else 0
            results[env_name + '_grasp_reward'] = np.mean(self.task_wise_grasp_reward[env_name]) if len(self.task_wise_grasp_reward[env_name]) > 0 else 0
            results[env_name + '_unscaled_reward'] = np.mean(self.task_wise_unscaled_reward[env_name]) if len(self.task_wise_unscaled_reward[env_name]) > 0 else 0
            results[env_name + '_TimeLimitTruncated'] = np.mean(self.task_wise_TimeLimitTruncated[env_name]) if len(self.task_wise_TimeLimitTruncated[env_name]) > 0 else 0
            results[env_name + '_grasp_success'] = np.mean(self.task_wise_grasp_success[env_name]) if len(self.task_wise_grasp_success[env_name]) > 0 else 0
            results[env_name + '_obj_to_target'] = np.mean(self.task_wise_obj_to_target[env_name]) if len(self.task_wise_obj_to_target[env_name]) > 0 else 0
            results[env_name + '_in_place_reward'] = np.mean(self.task_wise_in_place_reward[env_name]) if len(self.task_wise_in_place_reward[env_name]) > 0 else 0


        return results
        
    def initialize_episode_stats(self):
        env_names = self.env.get_attr('env_names')[0]
        self.task_wise_success = {}
        self.task_wise_near_object = {}
        self.task_wise_grasp_reward = {}
        self.task_wise_unscaled_reward = {}
        self.task_wise_TimeLimitTruncated = {}
        self.task_wise_grasp_success = {}
        self.task_wise_obj_to_target = {}
        self.task_wise_in_place_reward = {}
        for env_name in env_names:
            self.task_wise_success[env_name] = deque(maxlen=100)
            self.task_wise_near_object[env_name] = deque(maxlen=100)
            self.task_wise_grasp_reward[env_name] = deque(maxlen=100)
            self.task_wise_unscaled_reward[env_name] = deque(maxlen=100)
            self.task_wise_TimeLimitTruncated[env_name] = deque(maxlen=100)
            self.task_wise_grasp_success[env_name] = deque(maxlen=100)
            self.task_wise_obj_to_target[env_name] = deque(maxlen=100)
            self.task_wise_in_place_reward[env_name] = deque(maxlen=100)         

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


def make_parallel_metaworld_env(env_name, cfg, env_config, render_mode):
    is_obs_dict = True
    def make_env(rank, is_obs_dict):
        def _init():
            env = RandomizedMTEnv(
                eval=cfg.eval_policy, 
                # render_mode = None if cfg.no_render else "human", 
                render_mode=render_mode,
                is_obs_dict = is_obs_dict,
                env_name=cfg.mt_task
                )
            return env
        return _init
    if not cfg.eval_policy:
        return TorchWrapper(SubprocVecEnv([make_env(i, is_obs_dict = True) for i in range(cfg.env_agents)]), num_agents = cfg.env_agents, is_obs_dict = True)
    else:
        return TorchWrapper(SubprocVecEnv([make_env(i, is_obs_dict = True) for i in range(1)]), num_agents = 1, is_obs_dict = True)
