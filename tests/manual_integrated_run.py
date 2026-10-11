import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from coupling.engine import SimulationEngine
from spatial.config import load_config


config = load_config(
    "configs/default.yaml"
)

engine = SimulationEngine(
    config
)


print()
print("BEFORE")
engine.describe()


print()
print("RUNNING 0.1 BIOLOGICAL SECONDS...")
print()


engine.run(
    biological_seconds=0.1
)


engine.validate()


print()
print("AFTER")
engine.describe()
