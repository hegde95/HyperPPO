import sys
from sample_factory.enjoy import *
from sf_examples.swarm.train_swarm import parse_swarm_cfg, register_swarm_components, QUAD_BASELINE_CLI, override_params_for_batched_sampling, make_parallel_quadrotor_env
from sample_factory.envs.env_utils import register_env
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from typing import Any
from hyper.ghn_modules import MLP_GHN, MlpNetwork
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
from tqdm import tqdm
import pandas as pd


# QUADS_MODE_LIST_SINGLE = ['static_same_goal', 'static_diff_goal',  # static formations
#                           'ep_lissajous3D', 'ep_rand_bezier',  # evader pursuit
#                           'dynamic_same_goal',  # dynamic formations
#                           ]

# QUADS_MODE_LIST_SINGLE = ['static_same_goal']
QUADS_MODE_LIST_SINGLE = ['ep_rand_bezier']

def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False

    cfg = load_from_checkpoint(cfg)

    cfg.quads_view_mode = "local"
    # cfg.quads_view_mode = ["topdown", "chase", "global"]


    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} mmilestoneust be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1
    cfg.env_agents = 8*3
    # cfg.quads_render = True
    cfg.max_num_episodes = 40
    cfg.eval_deterministic = True
    num = 350

    # render_mode = "human"
    render_mode = None
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")


    if cfg.milestone_name is None:
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        milestone_to_load = checkpoint_dict["model"]
    else:
        milestone_dir = os.path.join(cfg.train_dir, cfg.experiment, "checkpoint_p0", "milestones")
        milestones = Learner.get_checkpoints(milestone_dir, "checkpoint_*")
        milestone_to_load = [milestone for milestone in milestones if cfg.milestone_name in milestone][0]
        checkpoint_dict = Learner.load_checkpoint([milestone_to_load], device)

    actor_critic.load_state_dict(checkpoint_dict["model"])
    ghn = actor_critic.actor_encoder.ghn
    ghn.eval()    

    experiment_dir = os.path.join(cfg.train_dir, cfg.experiment)
    dataframe_dir = os.path.join(experiment_dir, "dataframes")
    if not os.path.exists(dataframe_dir):
        os.makedirs(dataframe_dir)


    for quad_mode in QUADS_MODE_LIST_SINGLE:
        cfg.quads_mode = quad_mode
        env = make_env_func_batched(
            cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
        )
        env_info = extract_env_info(env, cfg)

        if hasattr(env.unwrapped, "reset_on_init"):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False


        list_of_test_archs = actor_critic.actor_encoder.list_of_arcs
        list_of_test_arch_indices = actor_critic.actor_encoder.list_of_arc_indices
        # self.list_of_test_shape_inds = self.actor_critic.actor_encoder.list_of_shape_inds
        list_of_test_shape_inds = torch.stack([actor_critic.actor_encoder.list_of_shape_inds[index][0:11] for k,index in enumerate(list_of_test_arch_indices)])
        test_results_df = pd.DataFrame(columns=['arch', 'num_params','reward'])
        test_results_df['arch'] = [list_of_test_archs[i] for i in list_of_test_arch_indices]
        test_results_df['num_params'] = [actor_critic.actor_encoder.get_params(list_of_test_archs[i]) for i in list_of_test_arch_indices]


        archs_per_num = len(list_of_test_arch_indices) // num  
        episode_rewards_means = np.zeros((len(list_of_test_arch_indices)))
        episode_rewards_stds = np.zeros((len(list_of_test_arch_indices)))

        for k in tqdm(range(num)):
            actor_critic.actor_encoder.set_graph(list_of_test_arch_indices[k*archs_per_num:(k+1)*archs_per_num], list_of_test_shape_inds[k*archs_per_num:(k+1)*archs_per_num])
            episode_rewards_per_num, finished_episode_per_num = eval_for_given_arch(env.num_agents, env, actor_critic, env_info, cfg, device)
            episode_rewards_means[k*archs_per_num:(k+1)*archs_per_num] = episode_rewards_per_num.reshape(archs_per_num, env.num_agents//archs_per_num).mean(1)
            episode_rewards_stds[k*archs_per_num:(k+1)*archs_per_num] = episode_rewards_per_num.reshape(archs_per_num, env.num_agents//archs_per_num).std(1)

        test_results_df['reward'] = episode_rewards_means
        test_results_df['reward_std'] = episode_rewards_stds
        dataframe_name = os.path.join(dataframe_dir, f"test_results_{quad_mode}_{cfg.milestone_name}.csv")
        test_results_df.to_csv(dataframe_name, index=False)
        print(f"Saved dataframe to {dataframe_name}")




    # return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
    #     [len(episode_rewards[i]) for i in range(env.num_agents)]
    # )
    return ExperimentStatus.SUCCESS, 0


def eval_for_given_arch(num_eval_envs, eval_env, actor_critic, env_info, cfg, device):
    episode_rewards = np.zeros((num_eval_envs))
    obs, infos = eval_env.reset()
    rnn_states = torch.zeros([num_eval_envs, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(num_eval_envs)]
    with torch.no_grad():
        while not all(finished_episode):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states, sample_actions=True)
            
            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)



            obs, rew, terminated, truncated, infos = eval_env.step(actions)
            dones = make_dones(terminated, truncated)
            infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            dones = dones.cpu().numpy()
            for agent_i, done_flag in enumerate(dones):
                if done_flag:
                    finished_episode[agent_i] = True
                    rew = episode_reward[agent_i].item()
                    episode_rewards[agent_i] = rew



                    episode_reward[agent_i] = 0

                    # reward_list.append(true_objective)
    return episode_rewards, finished_episode

class TorchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.observation_space.shape, dtype=np.float32)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)

    def step(self, action):
        # action = action.cpu().numpy()
        obs, reward, done, truncated, info = self.env.step(action)
        truncated = done
        obs = torch.from_numpy(obs).float().view(1,-1)
        reward = torch.tensor(reward).float().view(1,-1)
        done = torch.tensor(done).view(1,-1)
        truncated = torch.tensor(truncated).view(1,-1)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = torch.from_numpy(obs).float().view(1,-1)
        # obs_dict = {}
        # obs_dict['obs'] = obs
        # obs_dict['obs']['obs'] = obs
        info = torch.zeros(1)
        return obs, info
    
    def render(self):
        return self.env.render()

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
        obs, reward, done, truncated, info = self.env.step(action)
        return obs[0], reward[0], done[0], truncated[0], info[0]
    

def make_single_quadrotor_env(env_name, cfg, env_config, render_mode):
    return TorchWrapper(SimpleWrapper(make_quadrotor_env(env_name, cfg, _env_config = env_config, evaluation=True, render_mode = render_mode)))


def register_swarm_components(cfg):
    
    register_env("quadrotor_multi", make_parallel_quadrotor_env)
    if not cfg.hyper:
        register_models()

def main():
    """Script entry point."""
    cfg = parse_swarm_cfg(evaluation=True)
    register_swarm_components(cfg)
    enjoy(cfg)


if __name__ == '__main__':
    sys.exit(main())