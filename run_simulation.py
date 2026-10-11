import argparse

from coupling.engine import SimulationEngine
from spatial.config import load_config


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default="configs/default.yaml",
)

args = parser.parse_args()


config = load_config(
    args.config
)

engine = SimulationEngine(
    config
)

engine.describe()
