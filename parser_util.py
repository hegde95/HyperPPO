import argparse
import os
from distutils.util import strtobool


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

    parser.add_argument("--hyper", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use a Hyper network")
    parser.add_argument("--meta_batch_size", type=int, default=32,
        help="the number of meta batch size")
    parser.add_argument("--enable_arch_mixing", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Enable architecture mixing")
    parser.add_argument("--arch_conditional_critic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Enable architecture conditional critic")
    parser.add_argument("--dual_critic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Enable dual critic")
    parser.add_argument("--std_mode", type=str, default="multi",
        help="Get action std from either \
            (single: use same log_std vector for all architectures, \
             multi: use different log_std vector for each architecture, \
             arch_conditioned: Log std is conditioned on architecture with an MLP)")
    parser.add_argument("--multi_gpu", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Enable multi gpu training for the GHN. Enable this only if meta_batch_size is larger than 32 for a speedup, otherwise it will be slower.")
    parser.add_argument("--architecture_sampling", type=str, default="biased",
                        help="the architecture sampling method, has to be in [biased, uniform, sequential]")
    parser.add_argument("--num_episode_splits", type=int, default=8,
                        help="the number of architecture changes per episode")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v2",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=4_096_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4096,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
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
    parser.add_argument("--save-interval", type=int, default=25,
        help="Save interval")
    parser.add_argument("--run-name", type=str, default=None,
        help="Run name")
                                 
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    if args.hyper is False:
        args.arch_conditional_critic = False
        args.dual_critic = False
        args.enable_arch_mixing = False
    
    if args.multi_gpu:
        args.cuda = 0
    # fmt: on
    return args
