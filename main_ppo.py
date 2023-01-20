
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
# from lib.Model import ActorCritic, HyperActorCritic
from hyper.core import hyperActor


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=int, default=0,
        help="cuda will be enabled (run on cuda:0) by default, set to -1 to run on CPU, specify a positive int to run on a specific GPU")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="hyperppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--hyper", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use a Hyper network")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v2",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=2,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=125,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument('--wandb-tag', type=str,
        help='Use a custom tag for wandb. (default: "")')   
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Run in debug mode")     
                                 
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hyper = False):
        super().__init__()
        self.hyper = hyper
 
        if self.hyper:
            self.actor_mean = hyperActor(np.prod(envs.single_action_space.shape), np.array(envs.single_observation_space.shape).prod(), np.array([4,8,16,32,64,128,256,512]), meta_batch_size = 2, device=device)
            self.actor_mean.change_graph()

            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + self.actor_mean.arch_max_len, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )

        
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),                
                layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
            )

            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )      

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, ignore_value = False):
        if self.hyper:
            action_mean, _ = self.actor_mean(x)
            if ignore_value:
                value = None
            else:
                value = self.critic(torch.cat([self.actor_mean.arch_per_state_dim,x], -1))
        else:
            action_mean = self.actor_mean(x)
            value = self.critic(x)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def get_mean_action(self,x):
        if self.hyper:
            action_mean, _ = self.actor_mean(x)
        else:
            action_mean = self.actor_mean(x)
        
        return action_mean

def test_agent(envs, agent, device, num_episodes, hyper, max_steps = 1000):
    test_rewards = []

    for _ in range(num_episodes):
        obs = envs.reset()
        done = [False for _ in range(envs.num_envs)]
        episode_reward = 0
        for step in range(max_steps):
            obs = torch.FloatTensor(obs).to(device)
            action = agent.get_mean_action(obs)
            action = action.detach().cpu().numpy()
            obs, reward, done, _ = envs.step(action)
            episode_reward += np.mean(reward)
            if all(done):
                break
        test_rewards.append(episode_reward)
    return np.sum(test_rewards)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        tags = []
        if args.hyper:
            tags.append("hyper")
        else:
            tags.append("vanilla")
        if args.debug:
            tags.append("debug")                
        if args.wandb_tag:
            tags.append(args.wandb_tag)   

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and (args.cuda != -1) else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name, args.gamma) for i in range(args.num_envs)]
    )

    test_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + args.num_envs + i, i, args.capture_video, run_name, args.gamma) for i in range(8)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.hyper).to(device)
    if args.hyper:
        optimizer = torch.optim.Adam([
            {
                'params':agent.actor_mean.ghn.parameters(),
                'lr' :args.learning_rate, 
                # 'weight_decay' :1e-5,
            },
            {
                'params':agent.actor_logstd,
                'lr' :args.learning_rate,
                # 'weight_decay' :1e-5,
            },
            {
                'params':agent.critic.parameters(),
                'lr' :args.learning_rate,
            },
        ])
        
    else:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    if args.hyper:
        policy_shapes = torch.zeros((args.num_steps, args.num_envs) + (agent.actor_mean.arch_max_len,)).to(device)
        policy_shape_inds = -1 + torch.zeros((args.num_steps, args.num_envs) + (agent.actor_mean.shape_inds_max_len,)).to(device)
        policy_indices = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # if args.hyper:
    #     agent.actor_mean.change_graph()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.hyper:
            agent.actor_mean.change_graph()

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            if args.hyper:
                policy_shapes[step] = agent.actor_mean.arch_per_state_dim 
                policy_shape_inds[step] = agent.actor_mean.shape_ind_per_state_dim
                policy_indices[step] = agent.actor_mean.sampled_indices_per_state_dim

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # done = terminated | truncated
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # change the hyper network current model
            if args.hyper:
                agent.actor_mean.change_graph()

            if "episode" in info.keys():
                avg_ep_reward = np.mean([info['episode'][k]['r'] for k in range(args.num_envs)])
                avg_ep_length = np.mean([info['episode'][k]['l'] for k in range(args.num_envs)])
                print(f"global_step={global_step}, episodic_return={avg_ep_reward}")
                writer.add_scalar("charts/episodic_return", avg_ep_reward, global_step)
                writer.add_scalar("charts/episodic_length", avg_ep_length, global_step)

                # # change the hyper network current model
                # if args.hyper:
                #     agent.actor_mean.change_graph()

                break

                # for k in range(args.num_envs):
                #     print(f"global_step={global_step}, episodic_return={info['episode'][k]['r']}")
                #     writer.add_scalar("charts/episodic_return", info['episode'][k]["r"], global_step)
                #     writer.add_scalar("charts/episodic_length", info['episode'][k]["l"], global_step)
                #     break

            elif done.any():
                print(f"global_step={global_step}, episodic_return={rewards[:step+1,:].mean()}")
                writer.add_scalar("charts/episodic_return", rewards[:step+1,:].mean(), global_step)
        
        if args.hyper:
            final_policy_shape = agent.actor_mean.arch_per_state_dim 
            final_policy_shape_inds = agent.actor_mean.shape_ind_per_state_dim
            final_policy_indices = agent.actor_mean.sampled_indices_per_state_dim

        # bootstrap value if not done
        with torch.no_grad():
            if args.hyper:
                next_value = agent.critic(torch.cat([final_policy_shape ,next_obs],-1)).squeeze(-1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(step + 1)):
                if t == step:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs[:step+1].reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs[:step+1].reshape(-1)
        b_actions = actions[:step+1].reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages[:step+1].reshape(-1)
        b_returns = returns[:step+1].reshape(-1)
        b_values = values[:step+1].reshape(-1)
        
        if args.hyper:
            b_policy_shapes = policy_shapes[:step+1].reshape((-1,agent.actor_mean.arch_max_len))
            b_policy_shape_inds = policy_shape_inds[:step+1].reshape((-1,agent.actor_mean.shape_inds_max_len))
            b_policy_indices = policy_indices[:step+1].reshape(-1)

        # Optimizing the policy and value network
        batch_size = int(args.num_envs * (step + 1))
        minibatch_size = int(batch_size  // args.num_minibatches) 

        # make minibatch_size multiple of 8
        minibatch_size = minibatch_size - minibatch_size % 8
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if args.hyper:
                    # agent.actor_mean.change_graph(repeat_sample = False)
                    agent.actor_mean.set_graph(b_policy_shapes[mb_inds], b_policy_indices[mb_inds], b_policy_shape_inds[mb_inds])

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                _, newlogprob, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds], ignore_value = True)
                if args.hyper:
                    newvalue = agent.critic(torch.cat([b_policy_shapes[mb_inds], b_obs[mb_inds]], -1)).reshape(-1)
                else:
                    newvalue = agent.critic(b_obs[mb_inds]).reshape(-1)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        test_reward = test_agent(test_envs, agent, device, num_episodes=10, hyper=args.hyper, max_steps = 1000)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        print("test reward:", test_reward)
        writer.add_scalar("charts/test_reward", test_reward, global_step)

        print("------------------------------------------------------------")

    envs.close()
    writer.close()