import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import cupy as cp
import numpy as np


# ============================================================
# Make project root importable
# ============================================================

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


# ============================================================
# Configuration
# ============================================================

config = load_config(
    "configs/default.yaml"
)


# Keep this smoke test separate from normal simulation output.

smoke_output_dir = Path(
    "outputs/smoke_10s"
)


# Remove an old smoke-test directory so that output from a
# previous run cannot accidentally make this run look valid.

if smoke_output_dir.exists():

    shutil.rmtree(
        smoke_output_dir
    )


# Override only the output settings for this test.
#
# Analysis:
#     every 1 biological second
#
# Checkpoints:
#     every 5 biological seconds
#
# This means the 10-second run should produce roughly:
#
#     10 analysis snapshots
#     2 checkpoints

smoke_output_config = replace(
    config.output,

    directory=str(
        smoke_output_dir
    ),

    analysis_interval_s=1.0,

    checkpoint_interval_s=5.0,
)


config = replace(
    config,

    biological_duration_s=10.0,

    output=smoke_output_config,
)


# ============================================================
# Build engine
# ============================================================

print()
print("============================================")
print("10-SECOND END-TO-END SMOKE SIMULATION")
print("============================================")
print()


engine = SimulationEngine(
    config
)


engine.describe()


# ============================================================
# Record initial state
# ============================================================

initial_particles = (
    engine.particle_state
    .total_particle_count
)


initial_field_total = float(
    cp.sum(
        engine.field_state.values,
        dtype=cp.float64,
    ).item()
)


print()
print("INITIAL STATE")
print("--------------------------------------------")

print(
    f"Particles:                  "
    f"{initial_particles:,}"
)

print(
    f"Total field concentration:  "
    f"{initial_field_total:.6f}"
)


# ============================================================
# Run
# ============================================================

print()
print("RUNNING 10 BIOLOGICAL SECONDS...")
print()


cp.cuda.Stream.null.synchronize()

wall_start = time.perf_counter()


engine.run(
    biological_seconds=10.0
)


cp.cuda.Stream.null.synchronize()

wall_seconds = (
    time.perf_counter()
    - wall_start
)


# ============================================================
# Validate final state
# ============================================================

engine.validate()


final_particles = (
    engine.particle_state
    .total_particle_count
)


final_field_total = float(
    cp.sum(
        engine.field_state.values,
        dtype=cp.float64,
    ).item()
)


# ============================================================
# Verify output
# ============================================================

analysis_dir = (
    smoke_output_dir
    / "analysis"
)

checkpoint_dir = (
    smoke_output_dir
    / "checkpoints"
)


analysis_files = sorted(
    analysis_dir.glob(
        "analysis_tick_*.npz"
    )
)


checkpoint_paths = sorted(
    checkpoint_dir.glob(
        "checkpoint_tick_*"
    )
)


# ============================================================
# Reaction statistics
# ============================================================

reaction_events = (
    engine.reaction_engine
    .event_totals
)


# ============================================================
# Report
# ============================================================

print()
print("============================================")
print("FINAL RESULTS")
print("============================================")


print()
print("SIMULATION CLOCK")
print("--------------------------------------------")

print(
    f"Biological time:            "
    f"{engine.scheduler.current_time_s:.6f} s"
)

print(
    f"Master ticks:               "
    f"{engine.scheduler.current_tick:,}"
)

print(
    f"Wall time:                  "
    f"{wall_seconds:.3f} s"
)

print(
    f"Wall / biological second:   "
    f"{wall_seconds / 10.0:.3f} s"
)


print()
print("SUBSYSTEM UPDATES")
print("--------------------------------------------")

print(
    f"Particle diffusion steps:   "
    f"{engine.particle_diffusion.step_count:,}"
)

print(
    f"Field diffusion steps:      "
    f"{engine.field_diffusion.step_count:,}"
)

print(
    f"Reaction steps:             "
    f"{engine.reaction_engine.step_count:,}"
)


print()
print("PARTICLES")
print("--------------------------------------------")

print(
    f"Initial particles:          "
    f"{initial_particles:,}"
)

print(
    f"Final particles:            "
    f"{final_particles:,}"
)

print(
    f"Spill particles:            "
    f"{engine.particle_state.spill_particle_count:,}"
)

print(
    f"Spill writes:               "
    f"{engine.particle_diffusion.spill_write_count:,}"
)

print(
    f"Rejected moves:             "
    f"{engine.particle_diffusion.rejected_move_count:,}"
)

print(
    f"Hard overflow events:       "
    f"{engine.particle_diffusion.hard_overflow_count:,}"
)


print()
print("REACTIONS")
print("--------------------------------------------")

print(
    f"Degradation:                "
    f"{int(reaction_events[0]):,}"
)

print(
    f"Conversion:                 "
    f"{int(reaction_events[1]):,}"
)

print(
    f"Catalytic:                  "
    f"{int(reaction_events[2]):,}"
)

print(
    f"Binding:                    "
    f"{int(reaction_events[3]):,}"
)

print(
    f"Unbinding:                  "
    f"{int(reaction_events[4]):,}"
)

print(
    f"Particle-field evaluations: "
    f"{int(reaction_events[5]):,}"
)


print()
print("FIELDS")
print("--------------------------------------------")

print(
    f"Initial total:              "
    f"{initial_field_total:.6f}"
)

print(
    f"Final total:                "
    f"{final_field_total:.6f}"
)

print(
    f"Difference:                 "
    f"{final_field_total - initial_field_total:.8f}"
)


print()
print("OUTPUT")
print("--------------------------------------------")

print(
    f"Analysis snapshots:         "
    f"{len(analysis_files):,}"
)

print(
    f"Checkpoints:                "
    f"{len(checkpoint_paths):,}"
)

print(
    f"Output directory:           "
    f"{smoke_output_dir}"
)


# ============================================================
# Smoke-test assertions
# ============================================================

assert (
    engine.scheduler.current_time_s
    > 9.99
)


assert (
    engine.scheduler.current_time_s
    < 10.01
)


assert (
    engine.particle_diffusion.step_count
    > 0
)


assert (
    engine.field_diffusion.step_count
    > 0
)


assert (
    engine.reaction_engine.step_count
    > 0
)


assert (
    engine.particle_diffusion
    .hard_overflow_count
    == 0
)


assert (
    len(
        analysis_files
    )
    == 10
)


assert (
    len(
        checkpoint_paths
    )
    == 2
)


# Current demo field chemistry only transfers concentration
# between fields, so total field concentration should remain
# effectively conserved.

assert np.isclose(
    final_field_total,
    initial_field_total,
    rtol=1e-6,
    atol=1e-5,
)


print()
print("============================================")
print("SMOKE TEST PASSED")
print("============================================")
print()
