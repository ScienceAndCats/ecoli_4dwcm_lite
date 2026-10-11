from pyLM import *
from pyLM.units import *
import argparse
import logging
import time

parser = argparse.ArgumentParser()

parser.add_argument("--species", type=int, required=True)
parser.add_argument("--particles", type=int, default=100000)
parser.add_argument("-o", "--outputFile", required=True)

args = parser.parse_args()

LMLogger.setLMLogConsole(logging.WARNING)

NSPECIES = args.species
NPARTICLES = args.particles

print(f"Species:         {NSPECIES:,}")
print(f"Total particles:{NPARTICLES:,}")

total_start = time.perf_counter()

# ---------------------------------------------------------
# INTEGER RDME lattice
#
# Particle IDs use 32 bits instead of 8 bits.
# ---------------------------------------------------------

sim = RDME.IntRDMESimulation(
    dimensions=micron(3.2, 1.6, 1.6),
    spacing=nm(50)
)

# ---------------------------------------------------------
# Define species
# ---------------------------------------------------------

species = [
    f"S{i:05d}"
    for i in range(NSPECIES)
]

sim.defineSpecies(species)

region = sim.modifyRegion("default")
region.setDefaultDiffusionRate(1e-12)

# ---------------------------------------------------------
# Keep TOTAL particles constant at 100,000.
# Only the number of distinct identities changes.
# ---------------------------------------------------------

base_count = NPARTICLES // NSPECIES
remainder = NPARTICLES % NSPECIES

for i, sp in enumerate(species):

    count = base_count

    if i < remainder:
        count += 1

    if count > 0:
        sim.addParticles(
            species=sp,
            count=count
        )

# ---------------------------------------------------------
# Just ONE reaction.
#
# This keeps the reaction machinery active without making
# reaction-network complexity part of this benchmark.
# ---------------------------------------------------------

if NSPECIES >= 2:
    region.addReaction(
        reactant=species[0],
        product=species[1],
        rate=1.0
    )

# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

sim.setTimestep(ms(1))

# Write only at end
sim.setLatticeWriteInterval(1000)
sim.setWriteInterval(1000)

# 1000 timesteps = 1 biological second
sim.setSimulationTime(1.0)

build_done = time.perf_counter()

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

save_start = time.perf_counter()

sim.save(args.outputFile)

save_done = time.perf_counter()

# ---------------------------------------------------------
# GPU simulation
# ---------------------------------------------------------

run_start = time.perf_counter()

sim.run(
    filename=args.outputFile,
    method="lm::rdme::IntMpdRdmeSolver",
    replicates=1
)

run_done = time.perf_counter()

print()
print("TIMING")
print("-----------------------------------")
print(f"Model construction: {build_done-total_start:.3f} s")
print(f"Saving model:       {save_done-save_start:.3f} s")
print(f"GPU simulation:     {run_done-run_start:.3f} s")
print(f"Total Python time:  {run_done-total_start:.3f} s")
