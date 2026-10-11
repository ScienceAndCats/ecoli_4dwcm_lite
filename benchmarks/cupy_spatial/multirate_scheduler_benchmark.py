import argparse
import sys
import time
from pathlib import Path


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from compute.multirate import MultiRateScheduler


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--spacing-nm",
    type=float,
    default=50.0
)

parser.add_argument(
    "--D-fast",
    type=float,
    default=1.0,
    help="Fastest diffusion coefficient, um^2/s"
)

parser.add_argument(
    "--D-field",
    type=float,
    default=0.1,
    help="Representative slower field diffusion, um^2/s"
)

parser.add_argument(
    "--bio-seconds",
    type=float,
    default=1.0
)

parser.add_argument(
    "--reaction-ms",
    type=float,
    default=0.2969,
    help=(
        "Measured multifunction reaction "
        "kernel time in ms/update"
    )
)

args = parser.parse_args()


# ============================================================
# Base timestep derived from voxel size
# ============================================================

spacing_um = (
    args.spacing_nm
    / 1000.0
)

# Same conservative diffusion rule used so far.
q = 0.5 / 6.0

base_dt = (
    q
    * spacing_um**2
    / args.D_fast
)


# ============================================================
# Derive slower diffusion interval from its own D.
# ============================================================

field_dt = (
    q
    * spacing_um**2
    / args.D_field
)


# ============================================================
# Scheduler
# ============================================================

scheduler = MultiRateScheduler(
    base_dt_s=base_dt
)


# These callbacks are deliberately lightweight.
#
# This benchmark tests the SCHEDULER itself.
# Our GPU kernels have already been independently benchmarked.

def particle_diffusion(
    elapsed_dt,
    tick,
):
    pass


def reactions(
    elapsed_dt,
    tick,
):
    pass


def field_diffusion(
    elapsed_dt,
    tick,
):
    pass


def gene_expression(
    elapsed_dt,
    tick,
):
    pass


def slow_cell_state(
    elapsed_dt,
    tick,
):
    pass


# ============================================================
# Register tasks
# ============================================================

# Fastest process:
# every base timestep.

scheduler.add_task(
    name="particle_diffusion",
    interval_s=base_dt,
    callback=particle_diffusion,
)


# For this benchmark:
# request ~1 ms reaction updates.
#
# Scheduler converts this to the nearest integer number
# of base ticks.

scheduler.add_task(
    name="reactions",
    interval_s=0.001,
    callback=reactions,
)


# Slower diffusion interval is derived from:
#
# voxel size
# diffusion coefficient
#
# just like the fast timestep.

scheduler.add_task(
    name="field_diffusion",
    interval_s=field_dt,
    callback=field_diffusion,
)


# Representative slower biological processes.

scheduler.add_task(
    name="gene_expression",
    interval_s=0.010,
    callback=gene_expression,
)


scheduler.add_task(
    name="slow_cell_state",
    interval_s=0.100,
    callback=slow_cell_state,
)


# ============================================================
# Run scheduler
# ============================================================

start = time.perf_counter()

total_ticks = scheduler.run(
    args.bio_seconds
)

scheduler_wall = (
    time.perf_counter()
    - start
)


# ============================================================
# Compare with naive scheduling
# ============================================================

naive_calls = (
    total_ticks
    * len(scheduler.tasks)
)

multirate_calls = sum(
    task.calls
    for task in scheduler.tasks
)

call_reduction = (
    100.0
    * (
        1.0
        - multirate_calls
        / naive_calls
    )
)


# ============================================================
# Reaction workload projection
#
# Uses YOUR measured multifunction reaction time.
# ============================================================

reaction_task = next(
    task
    for task in scheduler.tasks
    if task.name == "reactions"
)

reaction_calls_multirate = (
    reaction_task.calls
)

reaction_calls_naive = (
    total_ticks
)

naive_reaction_wall_s = (
    reaction_calls_naive
    * args.reaction_ms
    / 1000.0
)

multirate_reaction_wall_s = (
    reaction_calls_multirate
    * args.reaction_ms
    / 1000.0
)


# Extrapolate reaction compute only to a
# 40-minute biological cell cycle.

scale_to_40min = (
    2400.0
    / args.bio_seconds
)

naive_40min_reaction_s = (
    naive_reaction_wall_s
    * scale_to_40min
)

multirate_40min_reaction_s = (
    multirate_reaction_wall_s
    * scale_to_40min
)


# ============================================================
# Output
# ============================================================

print()
print("BASE SIMULATION")
print("--------------------------------------------")

print(
    f"Voxel size:                 "
    f"{args.spacing_nm:.1f} nm"
)

print(
    f"Fastest D:                  "
    f"{args.D_fast:.3f} um^2/s"
)

print(
    f"Base timestep:              "
    f"{base_dt * 1e6:.2f} us"
)

print(
    f"Base ticks / bio second:    "
    f"{1.0/base_dt:,.0f}"
)

print(
    f"Benchmark biological time:  "
    f"{args.bio_seconds:.3f} s"
)

print(
    f"Total base ticks:           "
    f"{total_ticks:,}"
)


print()
print("TASK SCHEDULE")
print("--------------------------------------------")

for task in scheduler.tasks:

    error_us = (
        (
            task.actual_interval_s
            - task.requested_interval_s
        )
        * 1e6
    )

    print(
        f"{task.name:20s} "
        f"{task.interval_ticks:6d} ticks   "
        f"{task.actual_interval_s*1000:9.4f} ms   "
        f"{task.calls:7,d} calls   "
        f"error {error_us:+8.2f} us"
    )


print()
print("SCHEDULER")
print("--------------------------------------------")

print(
    f"Naive task calls:           "
    f"{naive_calls:,}"
)

print(
    f"Multirate task calls:       "
    f"{multirate_calls:,}"
)

print(
    f"Kernel-call reduction:      "
    f"{call_reduction:.1f}%"
)

print(
    f"Scheduler CPU wall time:    "
    f"{scheduler_wall*1000:.3f} ms"
)


print()
print("REACTION ENGINE PROJECTION")
print("--------------------------------------------")

print(
    f"Measured reaction/update:   "
    f"{args.reaction_ms:.4f} ms"
)

print(
    f"Naive reaction calls:       "
    f"{reaction_calls_naive:,}"
)

print(
    f"Multirate reaction calls:   "
    f"{reaction_calls_multirate:,}"
)

print(
    f"Naive reaction compute:     "
    f"{naive_reaction_wall_s:.3f} "
    f"wall s / benchmark"
)

print(
    f"Multirate reaction compute: "
    f"{multirate_reaction_wall_s:.3f} "
    f"wall s / benchmark"
)

print()
print("40-MIN CELL-CYCLE REACTION COMPUTE")
print("--------------------------------------------")

print(
    f"Naive:                      "
    f"{naive_40min_reaction_s/60:.2f} min"
)

print(
    f"Multirate:                  "
    f"{multirate_40min_reaction_s/60:.2f} min"
)
