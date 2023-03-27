import sys
import os

from sample_factory.enjoy import *
from sf_examples.brax.train_hyper_brax import parse_brax_cfg, register_brax_custom_components
from sample_factory.utils.wandb_utils import init_wandb
from tensorboardX import SummaryWriter

def enjoy(cfg: Config, following_train: bool = False) -> Tuple[StatusCode, float]:
    verbose = True

    if not following_train:
        cfg = load_from_checkpoint(cfg)
    cfg.env_agents = 8*8

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

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

    # policy_id = cfg.policy_index
    # name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    # checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    milestone_dir = os.path.join(cfg.train_dir, cfg.experiment, "checkpoint_p0", "milestones")
    milestones = Learner.get_checkpoints(milestone_dir, "checkpoint_*")
    summary_dir = os.path.join(cfg.train_dir, cfg.experiment, ".summary", "0")

    init_wandb(cfg)
    writer = SummaryWriter(summary_dir)
    
    for checkpoint in milestones:
        print(f"Evaluating:{checkpoint}")

        checkpoint_dict = Learner.load_checkpoint([checkpoint], device)
        actor_critic.load_state_dict(checkpoint_dict["model"])

        if cfg.hyper:
            list_of_test_archs = [
                [4, 4],
                [16],
                [16, 16, 16],
                [32, 32, 32],
                [64, 64, 64, 64],
                [128, 128, 128, 128],
                [256, 256, 256],
                [256, 256, 256, 256],
            ]
            list_of_test_arch_indices = [[i for i,arc in enumerate(actor_critic.actor_encoder.list_of_arcs) if list(arc) == t_arc][0] for t_arc in list_of_test_archs]
            list_of_test_shape_inds = torch.stack([actor_critic.actor_encoder.list_of_shape_inds[index][0:11] for k,index in enumerate(list_of_test_arch_indices)])
            actor_critic.actor_encoder.set_graph(list_of_test_arch_indices, list_of_test_shape_inds)


        episode_rewards = np.zeros((env.num_agents))
        num_frames = 0

        def max_frames_reached(frames):
            return cfg.max_num_frames is not None and frames > cfg.max_num_frames

        reward_list = []

        obs, infos = env.reset()
        rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
        episode_reward = None
        finished_episode = [False for _ in range(env.num_agents)]

        video_frames = []
        num_episodes = 0

        with torch.no_grad():
            while not all(finished_episode):
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

                if not cfg.no_render:
                    visualize_policy_inputs(normalized_obs)
                
                if cfg.hyper:
                    policy_outputs = actor_critic(normalized_obs, rnn_states, sample_actions=True)
                else:
                    policy_outputs = actor_critic(normalized_obs, rnn_states)

                # sample actions from the distribution by default
                actions = policy_outputs["actions"]

                # actions shape should be [num_agents, num_actions] even if it's [1, 1]
                if actions.ndim == 1:
                    actions = unsqueeze_tensor(actions, dim=-1)
                actions = preprocess_actions(env_info, actions)



                obs, rew, terminated, truncated, infos = env.step(actions)
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


            if all(finished_episode):
                if cfg.hyper:
                    average_reward_per_arch = episode_rewards.reshape(8,8).mean(1)
                    for i in range(len(list_of_test_arch_indices)):
                        print(f"test reward_{i}:", average_reward_per_arch[i])
                        writer.add_scalar(f"test_chart/{str(list_of_test_archs[i])}", average_reward_per_arch[i], checkpoint_dict['env_steps'])              
                    writer.add_scalar(f"test_chart/all_average", average_reward_per_arch.mean(), checkpoint_dict['env_steps'])
                else:
                    print(f"test reward:", episode_rewards.mean())
                    writer.add_scalar(f"test_chart/baseline_{cfg.encoder_mlp_layers}", episode_rewards.mean(), checkpoint_dict['env_steps'])
                    writer.add_scalar(f"test_chart/all_average", episode_rewards.mean(), checkpoint_dict['env_steps'])

    env.close()



    return ExperimentStatus.SUCCESS



def main():
    """Script entry point."""
    register_brax_custom_components(evaluation=False)
    cfg = parse_brax_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
