import sys
from pathlib import Path


# Make the project root importable when this file
# is run directly with:
#
# python tests/manual_particle_motion.py

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
engine.particle_state.describe()


engine.particle_diffusion.run_steps(
    100
)


print()
print("AFTER 100 DIFFUSION STEPS")
engine.particle_state.describe()

engine.particle_diffusion.describe()

engine.particle_diffusion.validate()