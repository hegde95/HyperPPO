from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.utils.algo_version import ALGO_VERSION

_params = ParamGrid(
    [
        ("seed", seeds(4)),
        ("env", ["humanoid"]),  #["ant", "humanoid", "halfcheetah", "walker2d"]
    ]
)

vstr = f"brax_hyper_humanoid_benchmark"

cli = "python -m sf_examples.brax.train_hyper_brax --with_wandb=True --wandb_tag hmn_0_vanilla --hyper False"

_experiments = [Experiment(vstr, cli, _params.generate_params())]
RUN_DESCRIPTION = RunDescription(vstr, experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.brax.experiments.brax_hyper_envs --backend=processes --max_parallel=4 --experiments_per_gpu=1 --num_gpus=4
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.brax.experiments.brax_hyper_envs --backend=slurm --slurm_workdir=./slurm_brax --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sf_examples/brax/experiments/sbatch_timeout_brax.sh --pause_between=1 --slurm_print_only=False
