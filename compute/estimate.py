import argparse
import math


# ============================================================
# Empirical calibration from SuperHans benchmarks
#
# These are intentionally kept visible and documented.
# Later we can move them into a YAML hardware profile.
# ============================================================

CALIBRATION = {
    "voxel_size_nm": 50.0,

    # 50-nm E. coli geometry
    "cell_voxels": 10792,

    # Integrated benchmark:
    # 500k particles
    # 12k possible species
    # 512 fields
    # 128 primary + 64 spill
    #
    # Measured:
    # 3.302 wall sec / biological sec
    "integrated_wall_per_bio_second": 3.302,

    "reference_particles": 500_000,
    "reference_fields": 512,

    # Output benchmarks
    "raw_checkpoint_512_mib": 25.11,
    "raw_checkpoint_2048_mib": 88.35,

    "analysis_snapshot_mib": 1.36,

    "raw_checkpoint_time_512_s": (
        0.00250 + 0.01337
    ),

    "raw_checkpoint_time_2048_s": (
        0.03344 + 0.03830
    ),

    "analysis_snapshot_time_s": (
        0.00037 + 0.00414
    ),
}


# ============================================================
# Geometry
# ============================================================

def estimate_cell_voxels(
    voxel_size_nm
):
    """
    Scale from measured 50-nm compact-cell voxel count.

    This is approximate because voxelization of the curved
    cell boundary is discrete.
    """

    ratio = (
        CALIBRATION["voxel_size_nm"]
        / voxel_size_nm
    )

    return round(
        CALIBRATION["cell_voxels"]
        * ratio**3
    )


# ============================================================
# Diffusion timestep
# ============================================================

def diffusion_dt(
    voxel_size_nm,
    diffusion_um2_s
):

    dx_um = (
        voxel_size_nm
        / 1000.0
    )

    q = (
        0.5
        / 6.0
    )

    return (
        q
        * dx_um**2
        / diffusion_um2_s
    )


# ============================================================
# Particle storage
# ============================================================

def particle_storage_mib(
    cell_voxels,
    primary_slots,
    spill_slots,
):

    # uint16 species IDs.
    #
    # Two buffers because diffusion ping-pongs.

    species_bytes = (
        cell_voxels
        * (
            primary_slots
            + spill_slots
        )
        * 2
        * 2
    )

    # primary + spill occupancies,
    # uint32, again with two buffers.

    occupancy_bytes = (
        cell_voxels
        * 2
        * 4
        * 2
    )

    return (
        species_bytes
        + occupancy_bytes
    ) / 1024**2


# ============================================================
# Concentration-field storage
# ============================================================

def field_storage_mib(
    cell_voxels,
    field_species,
    buffers=2,
):

    # float32

    return (
        cell_voxels
        * field_species
        * 4
        * buffers
        / 1024**2
    )


# ============================================================
# Checkpoint size
# ============================================================

def checkpoint_size_mib(
    cell_voxels,
    field_species,
    primary_slots,
    spill_slots,
):

    # One state copy on disk, not ping-pong buffers.

    particle_bytes = (
        cell_voxels
        * (
            primary_slots
            + spill_slots
        )
        * 2
    )

    occupancy_bytes = (
        cell_voxels
        * 2
        * 4
    )

    field_bytes = (
        cell_voxels
        * field_species
        * 4
    )

    return (
        particle_bytes
        + occupancy_bytes
        + field_bytes
    ) / 1024**2


# ============================================================
# Runtime
# ============================================================

def estimate_runtime(
    voxel_size_nm,
    particles,
    field_species,
    biological_seconds,
):
    """
    Rough empirical estimator.

    Calibrated to the integrated 50-nm benchmark.

    This is NOT intended to accurately extrapolate far from
    the calibrated model yet.

    Particle work is scaled approximately with particle count.

    Field work is scaled approximately with:
        voxel count * number of fields

    Diffusion-call frequency scales as dx^-2.
    """

    ref_voxel = (
        CALIBRATION[
            "voxel_size_nm"
        ]
    )

    ref_particles = (
        CALIBRATION[
            "reference_particles"
        ]
    )

    ref_fields = (
        CALIBRATION[
            "reference_fields"
        ]
    )

    ref_wall = (
        CALIBRATION[
            "integrated_wall_per_bio_second"
        ]
    )


    # Voxel count scaling

    voxel_ratio = (
        ref_voxel
        / voxel_size_nm
    )**3


    # Number of diffusion ticks per biological time.

    timestep_ratio = (
        ref_voxel
        / voxel_size_nm
    )**2


    particle_ratio = (
        particles
        / ref_particles
    )

    field_ratio = (
        field_species
        / ref_fields
    )


    # This intentionally gives particle and field work equal
    # weight for the first estimator.
    #
    # Once we accumulate more complete integrated benchmarks
    # we can fit these coefficients properly.

    state_work_ratio = (
        0.5 * particle_ratio
        +
        0.5 * (
            field_ratio
            * voxel_ratio
        )
    )


    estimated_wall_per_bio_second = (
        ref_wall
        * timestep_ratio
        * state_work_ratio
    )


    total_wall_seconds = (
        estimated_wall_per_bio_second
        * biological_seconds
    )

    return (
        estimated_wall_per_bio_second,
        total_wall_seconds
    )


# ============================================================
# Main
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--voxel-size-nm",
    type=float,
    default=50.0
)

parser.add_argument(
    "--particles",
    type=int,
    default=500000
)

parser.add_argument(
    "--particle-species",
    type=int,
    default=12000
)

parser.add_argument(
    "--field-species",
    type=int,
    default=512
)

parser.add_argument(
    "--primary-slots",
    type=int,
    default=128
)

parser.add_argument(
    "--spill-slots",
    type=int,
    default=64
)

parser.add_argument(
    "--D-fast",
    type=float,
    default=1.0
)

parser.add_argument(
    "--bio-seconds",
    type=float,
    default=2400.0
)

parser.add_argument(
    "--analysis-interval",
    type=float,
    default=1.0
)

parser.add_argument(
    "--checkpoint-interval",
    type=float,
    default=60.0
)

parser.add_argument(
    "--analysis-snapshot-mib",
    type=float,
    default=1.36
)

args = parser.parse_args()


# ============================================================
# Calculate
# ============================================================

cell_voxels = estimate_cell_voxels(
    args.voxel_size_nm
)

dt = diffusion_dt(
    args.voxel_size_nm,
    args.D_fast
)

ticks_per_second = (
    1.0 / dt
)


particle_memory = (
    particle_storage_mib(
        cell_voxels,
        args.primary_slots,
        args.spill_slots,
    )
)

field_memory = (
    field_storage_mib(
        cell_voxels,
        args.field_species,
    )
)

checkpoint_mib = (
    checkpoint_size_mib(
        cell_voxels,
        args.field_species,
        args.primary_slots,
        args.spill_slots,
    )
)


wall_per_bio_second, total_wall = (
    estimate_runtime(
        args.voxel_size_nm,
        args.particles,
        args.field_species,
        args.bio_seconds,
    )
)


analysis_snapshots = math.floor(
    args.bio_seconds
    / args.analysis_interval
)

checkpoint_snapshots = math.floor(
    args.bio_seconds
    / args.checkpoint_interval
)


analysis_storage_gib = (
    analysis_snapshots
    * args.analysis_snapshot_mib
    / 1024
)

checkpoint_storage_gib = (
    checkpoint_snapshots
    * checkpoint_mib
    / 1024
)


# ============================================================
# Output
# ============================================================

print()
print("SIMULATION ESTIMATE")
print("--------------------------------------------")

print(
    f"Voxel size:                 "
    f"{args.voxel_size_nm:.1f} nm"
)

print(
    f"Estimated cell voxels:      "
    f"{cell_voxels:,}"
)

print(
    f"Particles:                  "
    f"{args.particles:,}"
)

print(
    f"Particle species universe:  "
    f"{args.particle_species:,}"
)

print(
    f"Concentration fields:       "
    f"{args.field_species:,}"
)


print()
print("TIMING")
print("--------------------------------------------")

print(
    f"Fast diffusion timestep:    "
    f"{dt*1e6:.2f} us"
)

print(
    f"Base ticks / bio second:    "
    f"{ticks_per_second:,.0f}"
)

print(
    f"Biological duration:        "
    f"{args.bio_seconds:.1f} s"
)


print()
print("GPU STATE")
print("--------------------------------------------")

print(
    f"Particle buffers:           "
    f"{particle_memory:.2f} MiB"
)

print(
    f"Field buffers:              "
    f"{field_memory:.2f} MiB"
)

print(
    f"Estimated dynamic buffers:  "
    f"{particle_memory + field_memory:.2f} MiB"
)

print()
print(
    "Note: actual GPU usage will be higher because "
    "metadata, reaction tables, CuPy pools, temporary "
    "arrays, and CUDA state are not included."
)


print()
print("RUNTIME")
print("--------------------------------------------")

print(
    f"Estimated wall / bio sec:   "
    f"{wall_per_bio_second:.3f} s"
)

print(
    f"Estimated total wall time:  "
    f"{total_wall/3600:.2f} h"
)


print()
print("OUTPUT")
print("--------------------------------------------")

print(
    f"Raw checkpoint size:        "
    f"{checkpoint_mib:.2f} MiB"
)

print(
    f"Analysis interval:          "
    f"{args.analysis_interval:g} s"
)

print(
    f"Analysis snapshots:         "
    f"{analysis_snapshots:,}"
)

print(
    f"Analysis storage:           "
    f"{analysis_storage_gib:.2f} GiB"
)

print(
    f"Checkpoint interval:        "
    f"{args.checkpoint_interval:g} s"
)

print(
    f"Full checkpoints:           "
    f"{checkpoint_snapshots:,}"
)

print(
    f"Checkpoint storage:         "
    f"{checkpoint_storage_gib:.2f} GiB"
)

print(
    f"Total projected output:     "
    f"{analysis_storage_gib + checkpoint_storage_gib:.2f} GiB"
)


print()
print("CALIBRATION WARNING")
print("--------------------------------------------")

print(
    "Runtime is empirically calibrated to the 50-nm, "
    "500k-particle, 512-field integrated benchmark."
)

print(
    "Treat predictions far from that configuration as "
    "rough estimates until additional integrated benchmark "
    "points are available."
)
