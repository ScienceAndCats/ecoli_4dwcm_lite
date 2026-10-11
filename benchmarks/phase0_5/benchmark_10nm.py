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
# 10-nm E. coli-sized bounding lattice
#
# 3.2 x 1.6 x 1.6 µm
# 10-nm voxels
#
# 320 x 160 x 160
# = 8,192,000 lattice sites
# ---------------------------------------------------------

sim = RDME.RDMESimulation(
    dimensions=micron(3.2, 1.6, 1.6),
    spacing=nm(10)
)

sim.defineSpecies(["Protein"])

region = sim.modifyRegion("default")

# 1 µm²/s
region.setDefaultDiffusionRate(1e-12)

sim.addParticles(
    species="Protein",
    count=args.particles
)

# 50 microsecond timestep
sim.setTimestep(microsecond(50))

# In this GPU solver these effectively behave as timestep counts.
# 10,000 steps = 0.5 seconds biological time.
sim.setLatticeWriteInterval(10000)

# Species counts every 1000 steps = 0.05 biological seconds.
sim.setWriteInterval(1000)

sim.setSimulationTime(0.5)

sim.save(args.outputFile)

sim.run(
    filename=args.outputFile,
    method="lm::rdme::MpdRdmeSolver",
    replicates=1
)