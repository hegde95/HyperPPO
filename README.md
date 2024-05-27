# HyperPPO: A scalable method for finding small policies for robotic control
This repository consists of code used in this [paper: https://arxiv.org/abs/2309.16663](https://arxiv.org/abs/2309.16663). We present an algorithm that learns architecture-agnostic policies for RL for robotic tasks. This method enables us to find tiny neural networks, capable of modeling performant policies.

This work was accepted for oral presentation at the 2024 IEEE International Conference on Robotics and Automation (Yokohama). The paper website and video results can be found at [https://sites.google.com/usc.edu/hyperppo](https://sites.google.com/usc.edu/hyperppo)


This repo is based on 
```
https://github.com/alex-petrenko/sample-factory
```

To create env:
```
conda create -n hyper python==3.9

conda activate hyper

git clone git@github.com:alex-petrenko/sample-factory.git

cd sample-factory

pip install -e .

pip install chex==0.1.6

pip install flax==0.6.4

pip install orbax==0.1.1

pip install jax==0.3.25
```

Download the jax wheel file from https://drive.google.com/file/d/1dBwmHhFUe5bhBN3Zw48MzhXGhhDVL0sc/view?usp=sharing

```
pip install gdown

gdown https://drive.google.com/uc?id=1dBwmHhFUe5bhBN3Zw48MzhXGhhDVL0sc

pip install jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl 
```

Add the following line to .bashrc to avoid running into GPU memory issues:
```
echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> ~/.bashrc
```

To install stable-baselines3 (only for the vec env wrapper, for RL we use sample factory) and drone env:
```
git clone git@github.com:DLR-RM/stable-baselines3.git
cd stable-baselines3
pip install -e .
```
    
```
git clone git@github.com:Zhehui-Huang/quad-swarm-rl.git
cd quad-swarm-rl
pip install -e .
BEZIER_NO_EXTENSION=true python -m pip install bezier==2020.5.19
```

Remember to init wandb

Run:
```
python -m sample_factory.launcher.run --run=sf_examples.brax.experiments.brax_hyper_envs --backend=processes --max_parallel=4 --experiments_per_gpu=1 --num_gpus=4
```

Drone experiment:
```
python -m sf_examples.swarm.train_swarm --env quadrotor_multi --experiment hyper_test2 --train_dir dummy --train_for_env_steps 1_000_000_000 --dual_critic False --multi_stddev True --arch_sampling_mode biased --hyper True --with_wandb True --wandb_tags debug --meta_batch_size 16 --continuous_tanh_scale 15
```

# Citation
```
@article{hegde2023hyperppo,
  title={HyperPPO: A scalable method for finding small policies for robotic control},
  author={Hegde, Shashank and Huang, Zhehui and Sukhatme, Gaurav S},
  journal={arXiv preprint arXiv:2309.16663},
  year={2023}
}
```
