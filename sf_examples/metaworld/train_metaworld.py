import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.encoder import Encoder

from sf_examples.metaworld.metaworld_params import add_metaworld_env_args, metaworld_override_defaults
from sf_examples.metaworld.models import MTEncoder
from sf_examples.metaworld.metaworld_utils import make_parallel_metaworld_env


        

def make_mt_encoder(cfg, obs_space) -> Encoder:
    return MTEncoder(cfg, obs_space)

def register_models():
    global_model_factory().register_encoder_factory(make_mt_encoder)

def register_metaworld_components():
    
    register_env("metaworld_multi", make_parallel_metaworld_env)
    register_models()


def parse_metaworld_cfg(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_metaworld_env_args(partial_cfg.env, parser)
    metaworld_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_metaworld_components()
    cfg = parse_metaworld_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
# python -m sf_examples.metaworld.train_metaworld --env metaworld_multi --experiment bpt2 --train_dir dummy --hyper False --mt_task button-press-topdown --with_wandb True --wandb_group single_task --wandb_tags second