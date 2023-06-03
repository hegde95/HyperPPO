This repo is based on 
```
https://github.com/alex-petrenko/sample-factory
```

To create env:
```
conda create -n hyper python==3.9

git clone git@github.com:alex-petrenko/sample-factory.git

cd sample-factory

pip install -e .

pip install chex==0.1.6

pip install flax==0.6.4

pip install orbax==0.1.1

pip install jax==0.3.25

pip install jaxlib-0.3.25+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl 

```

Add the following line to .bashrc to avoid running into GPU memory issues:
```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Run:
```
python -m sample_factory.launcher.run --run=sf_examples.brax.experiments.brax_hyper_envs --backend=processes --max_parallel=4 --experiments_per_gpu=1 --num_gpus=4
```