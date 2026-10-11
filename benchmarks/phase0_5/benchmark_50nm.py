from pyLM import *
from pyLM.units import *
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outputFile", required=True)
parser.add_argument("-n", "--particles", type=int, default=1000)
args = parser.parse_args()

LMLogger.setLMLogConsole(logging.WARNING)

# ---------------------------------------------------------
# 50-nm E. coli-sized bounding lattice
#
# 3.2 x 1.6 x 1.6 µm
# 50-nm voxels
#
# = 64 x 32 x 32
# = 65,536 lattice sites
# ---------------------------------------------------------

sim = RDME.RDMESimulation(
    dimensions=micron(3.2, 1.6, 1.6),
    spacing=nm(50)
)

# One simple diffusing species.
sim.defineSpecies(["Protein"])

region = sim.modifyRegion("default")

# 1 µm²/s = 1e-12 m²/s
region.setDefaultDiffusionRate(1e-12)

# Keep particle number intentionally low for this benchmark.
# Particle-density scaling comes later.
sim.addParticles(
    species="Protein",
    count=args.particles
)

# 1-ms RDME timestep.
sim.setTimestep(ms(1))

# IMPORTANT:
# With this LM GPU solver these values effectively behave as
# timestep counts rather than seconds.
#
# 10,000 steps = 10 seconds -> only one full lattice output.
sim.setLatticeWriteInterval(10000)

# Species counts every 1000 steps = every 1 simulated second.
sim.setWriteInterval(1000)

sim.setSimulationTime(10.0)

sim.save(args.outputFile)

sim.run(
    filename=args.outputFile,
    method="lm::rdme::MpdRdmeSolver",
    replicates=1
)