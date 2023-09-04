from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.utils.algo_version import ALGO_VERSION

_params = ParamGrid(
    [
        ("seed", seeds(4)),
        ("env", ["quadrotor_multi"]),
        ("multi_stddev", [False]),
        ("dual_critic", [False]),
        ("arch_sampling_mode", ["biased"]),
        ("eval_every_steps", [0]),

    ]
)

vstr = f"swarm"

cli = "python -m sf_examples.swarm.train_swarm --with_wandb=True --wandb_tag paper_bench_swarm_neweval --hyper False --wandb_user khegde --wandb_group drones_actual_neweval --train_for_env_steps 1_000_000_000 --meta_batch_size 16 --continuous_tanh_scale 5"

_experiments = [Experiment(vstr, cli, _params.generate_params())]
RUN_DESCRIPTION = RunDescription(vstr, experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.swarm.experiments.swarm_basic_envs --backend=processes --max_parallel=4 --experiments_per_gpu=1 --num_gpus=4
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.swarm.experiments.swarm_basic_envs --backend=slurm --slurm_workdir=./slurm_swarm --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sf_examples/swarm/experiments/sbatch_timeout_swarm.sh --pause_between=1 --slurm_print_only=False
