from pyLM import *
from pyLM.units import *
import argparse
import logging

parser = argparse.ArgumentParser()

parser.add_argument("--species", type=int, required=True)
parser.add_argument("--reactions", type=int, required=True)
parser.add_argument("--particles", type=int, default=100000)
parser.add_argument("-o", "--outputFile", required=True)

args = parser.parse_args()

LMLogger.setLMLogConsole(logging.WARNING)

NSPECIES = args.species
NREACTIONS = args.reactions
NPARTICLES = args.particles

print("Species:", NSPECIES)
print("Reactions:", NREACTIONS)
print("Total particles:", NPARTICLES)

# ---------------------------------------------------------
# Same 50-nm lattice as our previous benchmark
# 64 x 32 x 32 = 65,536 voxels
# ---------------------------------------------------------

sim = RDME.RDMESimulation(
    dimensions=micron(3.2, 1.6, 1.6),
    spacing=nm(50)
)

# ---------------------------------------------------------
# Species
# ---------------------------------------------------------

species = [
    "S%03d" % i
    for i in range(NSPECIES)
]

sim.defineSpecies(species)

region = sim.modifyRegion("default")

region.setDefaultDiffusionRate(1e-12)

# ---------------------------------------------------------
# Distribute 100,000 particles approximately evenly
# among all species.
# ---------------------------------------------------------

base_count = NPARTICLES // NSPECIES
remainder = NPARTICLES % NSPECIES

for i, sp in enumerate(species):

    count = base_count

    if i < remainder:
        count += 1

    sim.addParticles(
        species=sp,
        count=count
    )

# ---------------------------------------------------------
# Synthetic first-order reactions.
#
# They preserve total particle number:
#
#     S_i -> S_j
#
# As reaction count increases, each species simply gets
# more possible transitions.
# ---------------------------------------------------------

for r in range(NREACTIONS):

    reactant_index = r % NSPECIES

    # Cycle through different destination species as
    # reaction number increases.
    offset = 1 + ((r // NSPECIES) % max(1, NSPECIES - 1))

    product_index = (
        reactant_index + offset
    ) % NSPECIES

    # Avoid self reactions.
    if product_index == reactant_index:
        product_index = (
            product_index + 1
        ) % NSPECIES

    region.addReaction(
        reactant=species[reactant_index],
        product=species[product_index],
        rate=1.0
    )

# ---------------------------------------------------------
# Simulation
#
# 1 ms timestep
# 1000 timesteps = 1 biological second
#
# Only write at the end so I/O doesn't dominate.
# ---------------------------------------------------------

sim.setTimestep(ms(1))

sim.setLatticeWriteInterval(1000)
sim.setWriteInterval(1000)

sim.setSimulationTime(1.0)

sim.save(args.outputFile)

sim.run(
    filename=args.outputFile,
    method="lm::rdme::MpdRdmeSolver",
    replicates=1
)
