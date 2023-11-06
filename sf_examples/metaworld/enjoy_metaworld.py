import sys

from sample_factory.enjoy import enjoy
from sf_examples.metaworld.train_metaworld import parse_metaworld_cfg, register_metaworld_components


def main():
    """Script entry point."""
    register_metaworld_components()
    cfg = parse_metaworld_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
