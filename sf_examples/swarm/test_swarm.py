import sys
from sample_factory.enjoy import *
from sf_examples.swarm.train_swarm import parse_swarm_cfg, register_swarm_components, QUAD_BASELINE_CLI, override_params_for_batched_sampling
from sample_factory.envs.env_utils import register_env
from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from typing import Any
from hyper.ghn_modules import MLP_GHN, MlpNetwork
from stable_baselines3.common.vec_env import SubprocVecEnv
import os




def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = True

    list_of_test_archs = [
                    [32, 32],                                   
                ]
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
    cfg.env_agents = 1
    cfg.quads_render = True
    cfg.max_num_episodes = 40
    cfg.eval_deterministic = True

    render_mode = "human"
    # render_mode = None
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

    # test_arch_model = MlpNetwork(fc_layers=arch_to_test, inp_dim = env.observation_space['obs'].shape[0], out_dim = env.action_space.shape[0]).to(device=device)
    list_of_test_arch_indices = [[i for i,arc in enumerate(actor_critic.actor_encoder.list_of_arcs) if list(arc) == t_arc][0] for t_arc in list_of_test_archs]
    list_of_test_shape_inds = torch.stack([actor_critic.actor_encoder.list_of_shape_inds[index][0:11] for k,index in enumerate(list_of_test_arch_indices)])

    # arch_index = [i for i,arc in enumerate(actor_critic.actor_encoder.list_of_arcs) if list(arc) == arch_to_test][0]
    # shape_ind = actor_critic.actor_encoder.list_of_shape_inds[arch_index]
    # shape_ind = shape_ind[:torch.where(shape_ind == -1.0)[0][0]]

    # _ = ghn([test_arch_model], return_embeddings=False, shape_ind = shape_ind.view(-1,1))
    actor_critic.actor_encoder.set_graph(list_of_test_arch_indices, list_of_test_shape_inds)
    test_arch_policy = actor_critic.actor_encoder.current_model[0]

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0
    total_rewraw_spin = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)

            # policy_outputs = actor_critic(normalized_obs, rnn_states, sample_actions=True)
            # x, std_dev = actor_critic.forward_head(normalized_obs)
            # actions = actor_critic.actor_encoder(normalized_obs['obs'])[0]

            actions = test_arch_policy(normalized_obs['obs'])


            # x, new_rnn_states = actor_critic.forward_core(x, rnn_states)
            # policy_outputs = actor_critic.forward_tail(x, False, sample_actions=True, std_dev=std_dev)

            # actor_decoder_output = x[:,:4]
            # action_distribution_params, _ = actor_critic.action_parameterization(actor_decoder_output, action_stddevs = std_dev)

            # sample actions from the distribution by default
            # actions = policy_outputs["actions"]
            # actions = policy_outputs["action_logits"][:,:4]
            # actions = x

            # # actions = test_arch_model(normalized_obs['obs'])
            # action_distribution_params, action_distribution = actor_critic.action_parameterization(actions.view(1,-1), action_stddevs = None)

            # if cfg.eval_deterministic:
            #     # action_distribution = actor_critic.action_distribution()
            #     actions = argmax_actions(action_distribution)[0]

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            # rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                total_rewraw_spin += infos['rewards']['rewraw_spin']
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate([dones]):
                    if done_flag.all():
                        finished_episode[agent_i] = True
                        rew = [episode_reward][0][agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get("true_objective", rew)
                        true_objectives[agent_i].append(true_objective)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                true_objectives[agent_i][-1],
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        [episode_reward][agent_i] = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1
                            reward_list.append(true_objective)

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ", "
                            avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

                    log.info(
                        "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
                    )
                    log.info(
                        "Avg episode reward: %.3f, avg true_objective: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
                    )
                    log.info("Spin reward: %.3f", total_rewraw_spin)
                    total_rewraw_spin = 0

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    if cfg.save_video:
        if cfg.fps > 0:
            fps = cfg.fps
        else:
            fps = 30
        generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

    if cfg.push_to_hub:
        generate_model_card(
            experiment_dir(cfg=cfg),
            cfg.algo,
            cfg.env,
            cfg.hf_repository,
            reward_list,
            cfg.enjoy_script,
            cfg.train_script,
        )
        push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)

    return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(episode_rewards[i]) for i in range(env.num_agents)]
    )


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
    
    register_env("quadrotor_multi", make_single_quadrotor_env)
    if not cfg.hyper:
        register_models()

def main():
    """Script entry point."""
    cfg = parse_swarm_cfg(evaluation=True)
    register_swarm_components(cfg)
    enjoy(cfg)


if __name__ == '__main__':
    sys.exit(main())