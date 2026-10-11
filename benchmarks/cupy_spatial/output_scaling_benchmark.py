import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import cupy as cp


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--spacing-nm", type=float, default=50.0)
parser.add_argument("--species", type=int, default=12000)
parser.add_argument("--molecules", type=int, default=500000)

parser.add_argument("--primary-slots", type=int, default=128)
parser.add_argument("--spill-slots", type=int, default=64)

parser.add_argument("--field-species", type=int, default=512)

parser.add_argument(
    "--selected-fields",
    type=int,
    default=32,
    help="Fields saved in lightweight analysis output"
)

parser.add_argument(
    "--repeats",
    type=int,
    default=3
)

parser.add_argument(
    "--output-dir",
    default="output_scaling_tmp"
)

args = parser.parse_args()


# ============================================================
# Geometry
#
# Same cell geometry as our integrated benchmark.
# Everything derives from voxel size.
# ============================================================

spacing_um = args.spacing_nm / 1000.0

CELL_LENGTH = 2.4
CELL_DIAMETER = 0.9
RADIUS = CELL_DIAMETER / 2.0

HALF_CYLINDER = (
    CELL_LENGTH - CELL_DIAMETER
) / 2.0

NX = int(np.ceil(
    (CELL_LENGTH + 2 * spacing_um)
    / spacing_um
))

NY = int(np.ceil(
    (CELL_DIAMETER + 2 * spacing_um)
    / spacing_um
))

NZ = NY

x = (
    np.arange(NX)
    - (NX - 1) / 2
) * spacing_um

y = (
    np.arange(NY)
    - (NY - 1) / 2
) * spacing_um

z = (
    np.arange(NZ)
    - (NZ - 1) / 2
) * spacing_um

X, Y, Z = np.meshgrid(
    x,
    y,
    z,
    indexing="ij"
)

axial_excess = np.maximum(
    np.abs(X) - HALF_CYLINDER,
    0.0
)

mask = (
    axial_excess**2
    + Y**2
    + Z**2
) <= RADIUS**2

CELL_VOXELS = int(
    np.sum(mask)
)


# ============================================================
# Generate representative packed-particle state
# ============================================================

rng = np.random.default_rng(12345)

initial_voxels = rng.integers(
    0,
    CELL_VOXELS,
    size=args.molecules,
    dtype=np.int32
)

initial_species = rng.integers(
    1,
    args.species + 1,
    size=args.molecules,
    dtype=np.uint16
)

order = np.argsort(
    initial_voxels,
    kind="stable"
)

vox_sorted = initial_voxels[order]
species_sorted = initial_species[order]

total_occ = np.bincount(
    vox_sorted,
    minlength=CELL_VOXELS
).astype(np.uint32)

TOTAL_CAPACITY = (
    args.primary_slots
    + args.spill_slots
)

if int(total_occ.max()) > TOTAL_CAPACITY:

    raise RuntimeError(
        "Initial occupancy exceeds "
        "primary+spill capacity."
    )


primary_cpu = np.zeros(
    (
        CELL_VOXELS,
        args.primary_slots
    ),
    dtype=np.uint16
)

spill_cpu = np.zeros(
    (
        CELL_VOXELS,
        args.spill_slots
    ),
    dtype=np.uint16
)

primary_occ_cpu = np.zeros(
    CELL_VOXELS,
    dtype=np.uint32
)

spill_occ_cpu = np.zeros(
    CELL_VOXELS,
    dtype=np.uint32
)


starts = (
    np.cumsum(
        total_occ,
        dtype=np.int64
    )
    - total_occ
)


for voxel in range(CELL_VOXELS):

    n = int(
        total_occ[voxel]
    )

    if n == 0:
        continue

    start = int(
        starts[voxel]
    )

    end = start + n

    species_here = species_sorted[
        start:end
    ]

    n_primary = min(
        n,
        args.primary_slots
    )

    n_spill = (
        n - n_primary
    )

    primary_cpu[
        voxel,
        :n_primary
    ] = species_here[
        :n_primary
    ]

    primary_occ_cpu[voxel] = (
        n_primary
    )

    if n_spill:

        spill_cpu[
            voxel,
            :n_spill
        ] = species_here[
            n_primary:
        ]

        spill_occ_cpu[voxel] = (
            n_spill
        )


# ============================================================
# Put state on GPU
# ============================================================

primary_gpu = cp.asarray(
    primary_cpu
)

spill_gpu = cp.asarray(
    spill_cpu
)

primary_occ_gpu = cp.asarray(
    primary_occ_cpu
)

spill_occ_gpu = cp.asarray(
    spill_occ_cpu
)


# Make representative continuous fields.
#
# Random-ish values are conservative for compression because
# arbitrary float32 values generally do not compress extremely
# well.

fields_gpu = cp.random.random(
    (
        args.field_species,
        CELL_VOXELS
    )
).astype(cp.float32)


cp.cuda.Stream.null.synchronize()


# ============================================================
# Output directory
# ============================================================

output_root = Path(
    args.output_dir
)

if output_root.exists():
    shutil.rmtree(output_root)

output_root.mkdir(
    parents=True
)


# ============================================================
# Helpers
# ============================================================

def directory_size(path):

    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def mib(nbytes):

    return (
        nbytes
        / 1024**2
    )


def gib(nbytes):

    return (
        nbytes
        / 1024**3
    )


# ============================================================
# GPU -> CPU transfer benchmarks
# ============================================================

def transfer_full():

    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    host = {
        "primary":
            cp.asnumpy(primary_gpu),

        "spill":
            cp.asnumpy(spill_gpu),

        "primary_occ":
            cp.asnumpy(primary_occ_gpu),

        "spill_occ":
            cp.asnumpy(spill_occ_gpu),

        "fields":
            cp.asnumpy(fields_gpu),
    }

    elapsed = (
        time.perf_counter()
        - start
    )

    return host, elapsed


selected_count = min(
    args.selected_fields,
    args.field_species
)


def transfer_analysis():

    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()

    # Lightweight output:
    # selected fields plus total voxel occupancy.

    selected = cp.asnumpy(
        fields_gpu[
            :selected_count
        ]
    )

    occupancy = cp.asnumpy(
        primary_occ_gpu
        + spill_occ_gpu
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "fields":
            selected,

        "occupancy":
            occupancy,
    }, elapsed


# ============================================================
# Writers
# ============================================================

def write_npy_directory(
    host,
    path
):

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True
    )

    start = time.perf_counter()

    for name, array in host.items():

        np.save(
            path / f"{name}.npy",
            array
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    size = directory_size(
        path
    )

    return elapsed, size


def write_npz(
    host,
    path,
    compressed=False
):

    start = time.perf_counter()

    if compressed:

        np.savez_compressed(
            path,
            **host
        )

    else:

        np.savez(
            path,
            **host
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    size = path.stat().st_size

    return elapsed, size


# ============================================================
# Warmup transfer
# ============================================================

_, _ = transfer_full()
_, _ = transfer_analysis()


# ============================================================
# FULL CHECKPOINT BENCHMARK
# ============================================================

full_transfer_times = []
raw_write_times = []
npz_write_times = []
compressed_write_times = []

raw_size = None
npz_size = None
compressed_size = None


for repeat in range(
    args.repeats
):

    host, transfer_time = (
        transfer_full()
    )

    full_transfer_times.append(
        transfer_time
    )


    raw_time, raw_size = (
        write_npy_directory(
            host,
            output_root / "full_raw"
        )
    )

    raw_write_times.append(
        raw_time
    )


    npz_time, npz_size = (
        write_npz(
            host,
            output_root / "full.npz",
            compressed=False
        )
    )

    npz_write_times.append(
        npz_time
    )


    compressed_time, compressed_size = (
        write_npz(
            host,
            output_root / "full_compressed.npz",
            compressed=True
        )
    )

    compressed_write_times.append(
        compressed_time
    )


# ============================================================
# LIGHTWEIGHT ANALYSIS OUTPUT
# ============================================================

analysis_transfer_times = []
analysis_write_times = []

analysis_size = None


for repeat in range(
    args.repeats
):

    host, transfer_time = (
        transfer_analysis()
    )

    analysis_transfer_times.append(
        transfer_time
    )


    write_time, analysis_size = (
        write_npz(
            host,
            output_root / "analysis.npz",
            compressed=False
        )
    )

    analysis_write_times.append(
        write_time
    )


# ============================================================
# Means
# ============================================================

full_transfer_mean = float(
    np.mean(
        full_transfer_times
    )
)

raw_write_mean = float(
    np.mean(
        raw_write_times
    )
)

npz_write_mean = float(
    np.mean(
        npz_write_times
    )
)

compressed_write_mean = float(
    np.mean(
        compressed_write_times
    )
)

analysis_transfer_mean = float(
    np.mean(
        analysis_transfer_times
    )
)

analysis_write_mean = float(
    np.mean(
        analysis_write_times
    )
)


# ============================================================
# Projections
# ============================================================

BIOLOGICAL_CELL_CYCLE = 2400.0


def snapshots_for_interval(
    interval_s
):

    return int(
        BIOLOGICAL_CELL_CYCLE
        / interval_s
    )


intervals = [
    1.0,
    10.0,
    60.0,
]


# ============================================================
# State sizes
# ============================================================

particle_state_bytes = (
    primary_gpu.nbytes
    + spill_gpu.nbytes
    + primary_occ_gpu.nbytes
    + spill_occ_gpu.nbytes
)

field_state_bytes = (
    fields_gpu.nbytes
)

full_raw_bytes = (
    particle_state_bytes
    + field_state_bytes
)


# ============================================================
# Output
# ============================================================

print()
print("CONFIGURATION")
print("--------------------------------------------")

print(
    f"Voxel size:                 "
    f"{args.spacing_nm:.1f} nm"
)

print(
    f"Cell voxels:                "
    f"{CELL_VOXELS:,}"
)

print(
    f"Particle species universe:  "
    f"{args.species:,}"
)

print(
    f"Particles represented:      "
    f"{args.molecules:,}"
)

print(
    f"Primary slots / voxel:      "
    f"{args.primary_slots}"
)

print(
    f"Spill slots / voxel:        "
    f"{args.spill_slots}"
)

print(
    f"Concentration fields:       "
    f"{args.field_species:,}"
)

print(
    f"Selected analysis fields:   "
    f"{selected_count:,}"
)

print(
    f"Benchmark repeats:          "
    f"{args.repeats}"
)


print()
print("IN-MEMORY STATE SIZE")
print("--------------------------------------------")

print(
    f"Particle state:             "
    f"{mib(particle_state_bytes):.2f} MiB"
)

print(
    f"Field state:                "
    f"{mib(field_state_bytes):.2f} MiB"
)

print(
    f"Full dynamic state:         "
    f"{mib(full_raw_bytes):.2f} MiB"
)


print()
print("FULL CHECKPOINT")
print("--------------------------------------------")

print(
    f"GPU -> CPU transfer:        "
    f"{full_transfer_mean*1000:.2f} ms"
)

print()

print(
    f"Raw .npy write:             "
    f"{raw_write_mean*1000:.2f} ms"
)

print(
    f"Raw .npy size:              "
    f"{mib(raw_size):.2f} MiB"
)

print()

print(
    f"Uncompressed .npz write:    "
    f"{npz_write_mean*1000:.2f} ms"
)

print(
    f"Uncompressed .npz size:     "
    f"{mib(npz_size):.2f} MiB"
)

print()

print(
    f"Compressed .npz write:      "
    f"{compressed_write_mean*1000:.2f} ms"
)

print(
    f"Compressed .npz size:       "
    f"{mib(compressed_size):.2f} MiB"
)


compression_ratio = (
    compressed_size
    / npz_size
)

print(
    f"Compressed / raw ratio:     "
    f"{compression_ratio:.3f}"
)


print()
print("LIGHTWEIGHT ANALYSIS OUTPUT")
print("--------------------------------------------")

print(
    f"Selected fields:            "
    f"{selected_count}"
)

print(
    f"GPU -> CPU transfer:        "
    f"{analysis_transfer_mean*1000:.2f} ms"
)

print(
    f"Write time:                 "
    f"{analysis_write_mean*1000:.2f} ms"
)

print(
    f"Snapshot size:              "
    f"{mib(analysis_size):.2f} MiB"
)


print()
print("40-MIN CELL-CYCLE STORAGE")
print("--------------------------------------------")

for interval in intervals:

    count = snapshots_for_interval(
        interval
    )

    raw_total = (
        raw_size
        * count
    )

    compressed_total = (
        compressed_size
        * count
    )

    analysis_total = (
        analysis_size
        * count
    )

    print()
    print(
        f"Every {interval:g} biological seconds:"
    )

    print(
        f"  snapshots:                "
        f"{count:,}"
    )

    print(
        f"  full raw:                 "
        f"{gib(raw_total):.2f} GiB"
    )

    print(
        f"  full compressed:          "
        f"{gib(compressed_total):.2f} GiB"
    )

    print(
        f"  lightweight analysis:     "
        f"{gib(analysis_total):.2f} GiB"
    )


print()
print("40-MIN OUTPUT WALL-TIME PROJECTION")
print("--------------------------------------------")

for interval in intervals:

    count = snapshots_for_interval(
        interval
    )

    raw_time = (
        full_transfer_mean
        + raw_write_mean
    ) * count

    compressed_time = (
        full_transfer_mean
        + compressed_write_mean
    ) * count

    analysis_time = (
        analysis_transfer_mean
        + analysis_write_mean
    ) * count

    print()
    print(
        f"Every {interval:g} biological seconds:"
    )

    print(
        f"  full raw:                 "
        f"{raw_time/60:.2f} min"
    )

    print(
        f"  full compressed:          "
        f"{compressed_time/60:.2f} min"
    )

    print(
        f"  lightweight analysis:     "
        f"{analysis_time/60:.2f} min"
    )


print()
print("NOTE")
print("--------------------------------------------")

print(
    "Full checkpoints are restartable state."
)

print(
    "Lightweight analysis output is not."
)

print(
    "Filesystem timings may benefit from OS caching, "
    "so use these mainly for relative comparison."
)
