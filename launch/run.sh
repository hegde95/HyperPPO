#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c10
#SBATCH --output=tmp/APL-%j.log

srun python main_ppo.py --hyper --True --track --wandb-tag bm1 --seed 111

# srun python main_ppo.py --hyper --True --track --wandb-tag bm1 --seed 222

# srun python main_ppo.py --hyper --True --track --wandb-tag bm1 --seed 333

# srun python main_ppo.py --hyper --True --track --wandb-tag bm1 --seed 444

# srun python main_ppo.py --hyper --True --track --wandb-tag bm1 --seed 555