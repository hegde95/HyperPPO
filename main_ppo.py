# code built on top of https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import warnings

# ignore UserWarnings and DeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import glob
import json
import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from parser_util import parse_args, argparse


def test_agent(envs, agent, device, num_episodes, hyper, max_steps = 1000, list_of_test_arch_indices = None, list_of_test_shape_inds = None, obs_normalizer = None):
    test_rewards = np.zeros((num_episodes, envs.num_envs))
    if obs_normalizer is not None:
        envs.obs_rms = obs_normalizer
    if hyper:
        # agent.actor_mean.change_graph()
        agent.actor_mean.set_graph(list_of_test_arch_indices, list_of_test_shape_inds)
    for ep in range(num_episodes):
        obs = envs.reset()
        done = [False for _ in range(envs.num_envs)]
        # episode_reward = np.zeros(envs.num_envs)
        for step in range(max_steps):
            obs = obs.to(device)
            action = agent.get_mean_action(obs)
            action = action.detach().cpu().numpy()
            obs, reward, done, info = envs.step(action)
            # episode_reward += reward
            # test_rewards[ep] += reward
            if all(done):
                break
        # test_rewards[ep] = np.array([info['episode'][k]['r'] for k in range(envs.num_envs)])
        test_rewards[ep] = envs.total_reward.cpu().numpy()
        # test_rewards.append(episode_reward)
    return test_rewards.mean(0).reshape(8,-1).mean(1)


if __name__ == "__main__":
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    from agent import Agent
    from envs.brax_custom.brax_env import make_vec_env_brax
    args.cuda = 0



    if args.run_name:
        run_name = args.run_name

        # load config json
        with open(f"runs/{run_name}/config.json") as f:
            config = json.load(f)
            args = argparse.Namespace(**config)
        
        args.run_name = run_name
        resume = True
        print(f"Resuming run: {run_name}") 

        with open(f"runs/{run_name}/stats.json") as f:
            stats = json.load(f)

        starting_update = stats["updates"]
        global_step = stats["global_step"]
        if args.track:
            wandb_id = stats["wandb_id"]

    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        resume = False
        print(f"Starting a new run: {run_name}")
        starting_update = 1
        global_step = 0
        if args.track:
            import wandb
            wandb_id = wandb.util.generate_id()

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
            id=wandb_id,
            resume="allow",
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

    envs = make_vec_env_brax("halfcheetah", args.num_envs, args.seed+1, torch.device("cuda:0"))  
    test_envs = make_vec_env_brax("halfcheetah", 8*3, args.seed, torch.device("cuda:0"))
    
    if resume:
        # load obs normalizer torch 
        envs.load_obs_rms(os.path.join('runs', run_name, 'obs_normalizer.pt'))
        test_envs.load_obs_rms(os.path.join('runs', run_name, 'test_obs_normalizer.pt'))

        # load return normalizer torch
        envs.load_return_rms(os.path.join('runs', run_name, 'return_normalizer.pt'))
        test_envs.load_return_rms(os.path.join('runs', run_name, 'test_return_normalizer.pt'))
        

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, device, args.hyper, meta_batch_size = args.meta_batch_size, arch_conditional_critic=args.arch_conditional_critic, \
                  state_conditioned_std=args.state_conditioned_std, dual_critic=args.dual_critic, multi_gpu = args.multi_gpu).to(device)
    if args.hyper:
        # optimizer = torch.optim.Adam([
        param_list = [
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
        ]
        if args.dual_critic:
            param_list.append({
                'params':agent.critic2.parameters(),
                'lr' :args.learning_rate,
            })
        optimizer = torch.optim.Adam(param_list, lr=args.learning_rate, eps=1e-5)
        
    else:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if resume:
        # agent.load_model(f"runs/{run_name}")
        # get all saved models
        model_files = glob.glob(f"runs/{run_name}/latest_model/*.pt")
        # get the last saved model
        model_file = sorted(model_files)[-1]
        # load the model
        agent.load_model(model_file)
        optimizer.load_state_dict(torch.load(f"runs/{run_name}/optimizer.pt"))

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    if args.hyper:
        # policy_shapes = torch.zeros((args.num_steps, args.num_envs) + (agent.actor_mean.arch_max_len,)).to(device)
        # policy_shape_inds = -1 + torch.zeros((args.num_steps, args.num_envs) + (agent.actor_mean.shape_inds_max_len,)).to(device)
        # policy_indices = torch.zeros((args.num_steps, args.num_envs)).to(device)

        if args.dual_critic:
            values2 = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs = envs.reset().to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

	
    # set test architectures as:
    # 0. [4, 4]
    # 1. [8, 8, 8]
    # 2. [16]
    # 3. [16, 16, 16]
    # 4. [32, 32, 32]
    # 5. [64, 64, 64, 64]
    # 6. [128, 128, 128, 128]
    # 7. [256, 256, 256, 256]
    if args.hyper:
        list_of_test_archs = [
            [4, 4],
            [8, 8, 8],
            [16],
            [16, 16, 16],
            [32, 32, 32],
            [64, 64, 64, 64],
            [128, 128, 128, 128],
            [256, 256, 256, 256],
        ]
        list_of_test_arch_indices = [[i for i,arc in enumerate(agent.actor_mean.list_of_arcs) if list(arc) == t_arc][0] for t_arc in list_of_test_archs]
        list_of_test_shape_inds = torch.stack([agent.actor_mean.list_of_shape_inds[index][0:11] for k,index in enumerate(list_of_test_arch_indices)])
    else:
        list_of_test_arch_indices = None
        list_of_test_shape_inds = None
        

    for update in range(starting_update, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            # next_obs = envs.reset().to(device)
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                if args.dual_critic:
                    value,value2 = value
                    values[step] = value.flatten()
                    values2[step] = value2.flatten()
                else:
                    values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # if args.hyper:
                # policy_shapes[step] = agent.actor_mean.arch_per_state_dim
                # policy_shape_inds[step] = agent.actor_mean.shape_ind_per_state_dim
                # policy_indices[step] = agent.actor_mean.sampled_indices_per_state_dim            

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # done = terminated | truncated
            rewards[step] = reward.to(device).view(-1)
            next_obs, next_done = next_obs.to(device), done.to(device)

            # if "episode" in info.keys():
            #     avg_ep_reward = np.mean([info['episode'][k]['r'] for k in range(args.num_envs)])
            #     avg_ep_length = np.mean([info['episode'][k]['l'] for k in range(args.num_envs)])
            #     print(f"global_step={global_step}, episodic_return={avg_ep_reward}")
            #     writer.add_scalar("charts/episodic_return", avg_ep_reward, global_step)
            #     writer.add_scalar("charts/episodic_length", avg_ep_length, global_step)


                # break

                # for k in range(args.num_envs):
                #     print(f"global_step={global_step}, episodic_return={info['episode'][k]['r']}")
                #     writer.add_scalar("charts/episodic_return", info['episode'][k]["r"], global_step)
                #     writer.add_scalar("charts/episodic_length", info['episode'][k]["l"], global_step)
                #     break

            if done.any():
                print(f"global_step={global_step}, episodic_return={envs.total_reward.cpu().numpy().mean()}")
                print(f"Time taken to accumulate this batch: {time.time() - start_time}")
                writer.add_scalar("charts/episodic_return", envs.total_reward.cpu().numpy().mean(), global_step)
                envs.total_reward = torch.zeros((envs.num_envs,)).to('cuda:0')
                break
        
        # if args.hyper:
        #     final_policy_shape = agent.actor_mean.arch_per_state_dim
        #     final_policy_shape_inds = agent.actor_mean.shape_ind_per_state_dim
        #     final_policy_indices = agent.actor_mean.sampled_indices_per_state_dim
        
        
        # bootstrap value if not done
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            if args.dual_critic:
                advantages2 = torch.zeros_like(rewards).to(device)
                lastgaelam2 = 0

                next_value, next_value2 = agent.get_value(next_obs)
                next_value = next_value.reshape(1, -1)
                next_value2 = next_value2.reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs).reshape(1, -1)
            for t in reversed(range(step + 1)):
                if t == step:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    if args.dual_critic:
                        nextvalues2 = next_value2
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    if args.dual_critic:
                        nextvalues2 = values2[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                if args.dual_critic:
                    delta2 = rewards[t] + args.gamma * nextvalues2 * nextnonterminal - values2[t]
                    advantages2[t] = lastgaelam2 = delta2 + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam2
                    
            returns = advantages + values
            if args.dual_critic:
                returns2 = advantages2 + values2

        # flatten the batch
        b_obs = obs[:step+1].swapaxes(0,1).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs[:step+1].swapaxes(0,1).reshape(-1)
        b_actions = actions[:step+1].swapaxes(0,1).reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages[:step+1].swapaxes(0,1).reshape(-1)
        b_returns = returns[:step+1].swapaxes(0,1).reshape(-1)
        b_values = values[:step+1].swapaxes(0,1).reshape(-1)

	
        if args.hyper:
            # b_policy_shapes = policy_shapes[:step+1].swapaxes(0,1).reshape((-1,agent.actor_mean.arch_max_len))
            # b_policy_shape_inds = policy_shape_inds[:step+1].swapaxes(0,1).reshape((-1,agent.actor_mean.shape_inds_max_len))
            # b_policy_indices = policy_indices[:step+1].reshape(-1)

            if args.dual_critic:
                b_advantages2 = advantages2[:step+1].swapaxes(0,1).reshape(-1)
                b_returns2 = returns2[:step+1].swapaxes(0,1).reshape(-1)
                b_values2 = values2[:step+1].swapaxes(0,1).reshape(-1)


            num_envs_per_arch = int(args.num_envs / args.meta_batch_size)

        # Optimizing the policy and value network
        if args.hyper and not args.enable_arch_mixing:
            batch_size = int(num_envs_per_arch * (step + 1))
        else:
            batch_size = int(args.num_envs * (step + 1))
        minibatch_size = int(batch_size  // args.num_minibatches)
        # make minibatch_size multiple of (meta_batch_size)
        minibatch_size = minibatch_size - minibatch_size % args.meta_batch_size
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if args.hyper and not args.enable_arch_mixing:
                    mb_inds = np.concatenate([b_inds[start:end] + k*batch_size for k in range(args.meta_batch_size)])
                else:
                    mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds] 

                if args.hyper:
                    # mb_policy_shapes = b_policy_shapes[mb_inds]
                    # mb_policy_shape_inds = b_policy_shape_inds[mb_inds]
                    # mb_policy_indices = b_policy_indices[mb_inds] 

                    if args.dual_critic:
                        mb_advantages2 = b_advantages2[mb_inds]
                        mb_returns2 = b_returns2[mb_inds]
                        mb_values2 = b_values2[mb_inds]

                if args.hyper:
                    agent.actor_mean.change_graph(repeat_sample = True)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                
                if args.hyper and args.dual_critic:
                    newvalue, newvalue2 = newvalue

                # if args.hyper:
                #     assert (agent.actor_mean.arch_per_state_dim == mb_policy_shapes).all(), "arch_per_state_dim != mb_policy_shapes"
                
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # mb_advantages = mb_advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    if args.dual_critic:
                        mb_advantages2 = (mb_advantages2 - mb_advantages2.mean()) / (mb_advantages2.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.dual_critic:
                    pg_loss2_1 = -mb_advantages2 * ratio
                    pg_loss2_2 = -mb_advantages2 * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss2 = torch.max(pg_loss2_1, pg_loss2_2).mean()

                    pg_loss_total = pg_loss + pg_loss2
                else:
                    pg_loss_total = pg_loss

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                if args.dual_critic:
                    newvalue2 = newvalue2.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped2 = (newvalue2 - mb_returns2) ** 2
                        v_clipped2 = mb_values2 + torch.clamp(
                            newvalue2 - mb_values2,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped2 = (v_clipped2 - mb_returns2) ** 2
                        v_loss_max2 = torch.max(v_loss_unclipped2, v_loss_clipped2)
                        v_loss2 = 0.5 * v_loss_max2.mean()
                    else:
                        v_loss2 = 0.5 * ((newvalue2 - mb_returns2) ** 2).mean()

                    v_loss_total = v_loss + v_loss2
                else:
                    v_loss_total = v_loss

                entropy_loss = entropy.mean()
                loss = pg_loss_total - args.ent_coef * entropy_loss + v_loss_total * args.vf_coef

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

        if args.hyper and args.dual_critic:
            y_pred2, y_true2 = b_values2.cpu().numpy(), b_returns2.cpu().numpy()
            var_y2 = np.var(y_true2)
            explained_var2 = np.nan if var_y2 == 0 else 1 - np.var(y_true2 - y_pred2) / var_y2

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if args.hyper and args.dual_critic:
            writer.add_scalar("losses/value_loss1", v_loss.item(), global_step)
            writer.add_scalar("losses/value_loss2", v_loss2.item(), global_step)

            writer.add_scalar("losses/policy_loss1", pg_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss2", pg_loss2.item(), global_step)

        writer.add_scalar("losses/value_loss", v_loss_total.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss_total.item(), global_step)

        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)

        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if args.hyper and args.dual_critic:
            writer.add_scalar("losses/explained_variance2", explained_var2, global_step)

        test_reward = test_agent(test_envs, agent, device, num_episodes=3, hyper=args.hyper, max_steps = 1000, list_of_test_arch_indices = list_of_test_arch_indices, list_of_test_shape_inds = list_of_test_shape_inds, obs_normalizer = envs.obs_rms)

	
        if args.hyper:              
            for i in range(len(list_of_test_arch_indices)):
                print(f"test reward_{i}:", test_reward[i])
                writer.add_scalar(f"charts/test_reward_{i}", test_reward[i], global_step)      
        print("test reward:", test_reward.mean())
        writer.add_scalar("charts/test_reward", test_reward.mean(), global_step)

        
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        agent.save_model(os.path.join('runs', run_name, 'latest_model'), -1)
        # save optimizer
        torch.save(optimizer.state_dict(), os.path.join(os.path.join('runs', run_name), "optimizer.pt"))
        # save args as a config file
        with open(os.path.join('runs', run_name, f"config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        with open(os.path.join('runs', run_name, f"stats.json"), "w") as f:
            json.dump({
                "updates":update, 
                "global_step":global_step,
                "wandb_id":wandb_id if args.track else None,
                }, f, indent=2)
        
        # save env obs normalizer and reward normalizer
        envs.save_obs_rms(os.path.join('runs', run_name, 'obs_normalizer.pt'))
        envs.save_return_rms(os.path.join('runs', run_name, 'return_normalizer.pt'))

        # save test env obs normalizer and reward normalizer
        test_envs.save_obs_rms(os.path.join('runs', run_name, 'test_obs_normalizer.pt'))
        test_envs.save_return_rms(os.path.join('runs', run_name, 'test_return_normalizer.pt'))

        if  update % args.save_interval == 0:
            agent.save_model(os.path.join('runs', run_name, 'checkpoints'), update)
            
        # change the hyper network current model
        if args.hyper:
            agent.actor_mean.change_graph(repeat_sample=False)

        print("------------------------------------------------------------")


    envs.close()
    writer.close()