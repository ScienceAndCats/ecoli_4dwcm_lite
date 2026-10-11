import argparse
import time
import numpy as np
import cupy as cp


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--spacing-nm", type=float, default=50.0)
parser.add_argument("--species", type=int, default=12000)
parser.add_argument("--molecules", type=int, default=250000)
parser.add_argument("--max-slots", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--D", type=float, default=1.0)

args = parser.parse_args()


# ============================================================
# Geometry
#
# IMPORTANT:
# Everything derives from voxel size.
# 50 nm is merely the default.
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
    x, y, z,
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

mask_flat = mask.reshape(-1)

BOUNDING_VOXELS = mask_flat.size

active_flat = np.flatnonzero(
    mask_flat
)

CELL_VOXELS = active_flat.size


# ============================================================
# Convert bounding-box voxel IDs -> compact cell voxel IDs
# ============================================================

flat_to_active = np.full(
    BOUNDING_VOXELS,
    -1,
    dtype=np.int32
)

flat_to_active[active_flat] = np.arange(
    CELL_VOXELS,
    dtype=np.int32
)


# ============================================================
# Compact neighbor table
#
# neighbors[cell_voxel, direction]
#
# If a molecule tries to leave the cell, it stays where it is.
# ============================================================

neighbors_cpu = np.empty(
    (CELL_VOXELS, 6),
    dtype=np.int32
)

plane = NY * NZ

for active_id, flat_id in enumerate(active_flat):

    xi = flat_id // plane

    rem = flat_id - xi * plane

    yi = rem // NZ
    zi = rem - yi * NZ

    candidates = [
        flat_id + plane if xi + 1 < NX else flat_id,
        flat_id - plane if xi > 0 else flat_id,
        flat_id + NZ if yi + 1 < NY else flat_id,
        flat_id - NZ if yi > 0 else flat_id,
        flat_id + 1 if zi + 1 < NZ else flat_id,
        flat_id - 1 if zi > 0 else flat_id,
    ]

    for direction, candidate in enumerate(candidates):

        candidate_active = flat_to_active[candidate]

        if candidate_active >= 0:
            neighbors_cpu[
                active_id,
                direction
            ] = candidate_active

        else:
            neighbors_cpu[
                active_id,
                direction
            ] = active_id


neighbors = cp.asarray(
    neighbors_cpu.reshape(-1)
)


# ============================================================
# Diffusion timestep derived from voxel size
# ============================================================

q = 0.5 / 6.0

dt = (
    q
    * spacing_um**2
    / args.D
)


# ============================================================
# Initialize packed particles on CPU
#
# species ID 0 = empty slot
#
# Actual species IDs therefore run:
#
# 1 ... args.species
# ============================================================

if args.species > 65534:
    raise ValueError(
        "uint16 species storage supports "
        "up to 65,534 biological species."
    )

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


# Sort molecules by voxel so we can pack them.
order = np.argsort(
    initial_voxels,
    kind="stable"
)

vox_sorted = initial_voxels[order]

species_sorted = initial_species[order]

occupancy_cpu = np.bincount(
    vox_sorted,
    minlength=CELL_VOXELS
).astype(np.uint32)

initial_max_occupancy = int(
    occupancy_cpu.max()
)

if initial_max_occupancy > args.max_slots:

    raise RuntimeError(
        f"Initial voxel occupancy "
        f"{initial_max_occupancy} exceeds "
        f"--max-slots {args.max_slots}. "
        f"Increase --max-slots."
    )


# Find each molecule's slot within its voxel.
starts = np.cumsum(
    occupancy_cpu,
    dtype=np.int64
) - occupancy_cpu

repeated_starts = np.repeat(
    starts,
    occupancy_cpu
)

slot_sorted = (
    np.arange(
        args.molecules,
        dtype=np.int64
    )
    - repeated_starts
)


packed_cpu = np.zeros(
    (CELL_VOXELS, args.max_slots),
    dtype=np.uint16
)

packed_cpu[
    vox_sorted,
    slot_sorted
] = species_sorted


# ============================================================
# CUDA
# ============================================================

CUDA_SRC = r'''

__device__ __forceinline__
unsigned long long mix64(
    unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ULL;

    x = (x ^ (x >> 30))
        * 0xbf58476d1ce4e5b9ULL;

    x = (x ^ (x >> 27))
        * 0x94d049bb133111ebULL;

    return x ^ (x >> 31);
}


__device__ __forceinline__
float random01(
    unsigned long long key)
{
    unsigned long long x = mix64(key);

    return
        ((x >> 40) & 0xFFFFFFULL)
        * (1.0f / 16777216.0f);
}


// ============================================================
// Packed stochastic diffusion
//
// Each thread corresponds to one possible voxel slot.
//
// Only occupied slots do work.
// ============================================================

extern "C" __global__
void packed_diffuse(
    const unsigned short* species_in,
    const unsigned int* occupancy_in,

    unsigned short* species_out,
    unsigned int* occupancy_out,

    const int* neighbors,

    unsigned int* overflow,

    int max_slots,
    long long nvox,

    float q,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    long long total =
        nvox
        * (long long)max_slots;

    if (idx >= total)
        return;

    int voxel =
        idx / max_slots;

    int slot =
        idx % max_slots;

    unsigned int n =
        occupancy_in[voxel];

    if (slot >= n)
        return;

    unsigned short species =
        species_in[
            (long long)voxel
            * max_slots
            + slot
        ];

    if (species == 0)
        return;


    unsigned long long key =
        ((unsigned long long)idx
         * 0x9e3779b97f4a7c15ULL)
        ^
        ((unsigned long long)step
         * 0xbf58476d1ce4e5b9ULL);

    float r =
        random01(key);

    int destination =
        voxel;

    if (r < 6.0f * q)
    {
        int direction =
            (int)(r / q);

        if (direction > 5)
            direction = 5;

        destination =
            neighbors[
                voxel * 6
                + direction
            ];
    }


    // Reserve one slot in the destination voxel.
    unsigned int destination_slot =
        atomicAdd(
            &occupancy_out[destination],
            1U
        );


    if (destination_slot < max_slots)
    {
        species_out[
            (long long)destination
            * max_slots
            + destination_slot
        ] = species;
    }

    else
    {
        // For benchmark purposes, count overflows.
        // A production simulator will use a spill buffer.
        atomicAdd(
            overflow,
            1U
        );
    }
}


// ============================================================
// Local voxel scan
//
// This represents the kind of operation our reaction engine
// will perform:
//
// "What classes of molecules are in this voxel?"
//
// No global sort is required.
// ============================================================

extern "C" __global__
void local_scan(
    const unsigned short* species,
    const unsigned int* occupancy,

    const unsigned char* reaction_class,

    unsigned int* result,

    int max_slots,
    long long nvox)
{
    long long voxel =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (voxel >= nvox)
        return;

    unsigned int n =
        occupancy[voxel];

    if (n > max_slots)
        n = max_slots;

    unsigned int class1 = 0;
    unsigned int class2 = 0;

    for (
        unsigned int slot = 0;
        slot < n;
        slot++
    )
    {
        unsigned short s =
            species[
                voxel
                * (long long)max_slots
                + slot
            ];

        unsigned char c =
            reaction_class[s];

        if (c == 1)
            class1++;

        else if (c == 2)
            class2++;
    }

    // Synthetic "reaction candidate" count.
    result[voxel] =
        class1 * class2;
}

'''


module = cp.RawModule(
    code=CUDA_SRC
)

packed_diffuse = module.get_function(
    "packed_diffuse"
)

local_scan = module.get_function(
    "local_scan"
)


# ============================================================
# GPU allocation
# ============================================================

allocation_start = time.perf_counter()

species_a = cp.asarray(
    packed_cpu.reshape(-1)
)

species_b = cp.empty_like(
    species_a
)

occupancy_a = cp.asarray(
    occupancy_cpu
)

occupancy_b = cp.zeros_like(
    occupancy_a
)

overflow = cp.zeros(
    1,
    dtype=cp.uint32
)


# Synthetic reaction class metadata.
#
# class 0 = ordinary
# class 1 = e.g. enzyme-like
# class 2 = e.g. substrate-like
#
# This is only for benchmarking local lookup.
reaction_class_cpu = np.zeros(
    args.species + 1,
    dtype=np.uint8
)

reaction_class_cpu[
    1::10
] = 1

reaction_class_cpu[
    2::10
] = 2

reaction_class = cp.asarray(
    reaction_class_cpu
)

scan_result = cp.zeros(
    CELL_VOXELS,
    dtype=cp.uint32
)

cp.cuda.Stream.null.synchronize()

allocation_time = (
    time.perf_counter()
    - allocation_start
)


# ============================================================
# Launch helpers
# ============================================================

THREADS = 256

TOTAL_SLOTS = (
    CELL_VOXELS
    * args.max_slots
)


def blocks(total):

    return (
        int(
            (total + THREADS - 1)
            // THREADS
        ),
    )


def gpu_time(function):

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    function()

    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(
        start,
        end
    )


# ============================================================
# Warmup
# ============================================================

occupancy_b.fill(0)
overflow.fill(0)

packed_diffuse(
    blocks(TOTAL_SLOTS),
    (THREADS,),
    (
        species_a,
        occupancy_a,

        species_b,
        occupancy_b,

        neighbors,

        overflow,

        np.int32(args.max_slots),
        np.int64(CELL_VOXELS),

        np.float32(q),
        np.uint64(0)
    )
)

local_scan(
    blocks(CELL_VOXELS),
    (THREADS,),
    (
        species_a,
        occupancy_a,
        reaction_class,
        scan_result,
        np.int32(args.max_slots),
        np.int64(CELL_VOXELS)
    )
)

cp.cuda.Stream.null.synchronize()


# Reset after warmup.
species_a = cp.asarray(
    packed_cpu.reshape(-1)
)

occupancy_a = cp.asarray(
    occupancy_cpu
)

species_b = cp.empty_like(
    species_a
)

occupancy_b = cp.zeros_like(
    occupancy_a
)

overflow.fill(0)


# ============================================================
# Diffusion benchmark
# ============================================================

def run_diffusion():

    global species_a
    global species_b
    global occupancy_a
    global occupancy_b

    for step in range(
        args.steps
    ):

        occupancy_b.fill(0)

        packed_diffuse(
            blocks(TOTAL_SLOTS),
            (THREADS,),
            (
                species_a,
                occupancy_a,

                species_b,
                occupancy_b,

                neighbors,

                overflow,

                np.int32(
                    args.max_slots
                ),

                np.int64(
                    CELL_VOXELS
                ),

                np.float32(q),

                np.uint64(step)
            )
        )

        species_a, species_b = (
            species_b,
            species_a
        )

        occupancy_a, occupancy_b = (
            occupancy_b,
            occupancy_a
        )


diffusion_ms = gpu_time(
    run_diffusion
)


# ============================================================
# Local reaction lookup benchmark
# ============================================================

def run_scans():

    for _ in range(
        args.steps
    ):

        local_scan(
            blocks(CELL_VOXELS),
            (THREADS,),
            (
                species_a,
                occupancy_a,
                reaction_class,
                scan_result,

                np.int32(
                    args.max_slots
                ),

                np.int64(
                    CELL_VOXELS
                )
            )
        )


scan_ms = gpu_time(
    run_scans
)


# ============================================================
# Results
# ============================================================

overflow_count = int(
    overflow[0]
)

final_molecules = int(
    cp.sum(
        cp.minimum(
            occupancy_a,
            args.max_slots
        )
    )
)

max_occupancy_final = int(
    cp.max(
        occupancy_a
    )
)

raw_particle_bytes = (
    species_a.nbytes
    + species_b.nbytes
    + occupancy_a.nbytes
    + occupancy_b.nbytes
)

neighbor_bytes = (
    neighbors.nbytes
)

pool = cp.get_default_memory_pool()


print()
print("CONFIGURATION")
print("------------------------------------------")

print(
    f"Voxel size:              "
    f"{args.spacing_nm:.1f} nm"
)

print(
    f"Bounding grid:           "
    f"{NX} x {NY} x {NZ}"
)

print(
    f"Bounding voxels:         "
    f"{BOUNDING_VOXELS:,}"
)

print(
    f"Active cell voxels:      "
    f"{CELL_VOXELS:,}"
)

print(
    f"Species universe:        "
    f"{args.species:,}"
)

print(
    f"Molecules:               "
    f"{args.molecules:,}"
)

print(
    f"Max slots / voxel:       "
    f"{args.max_slots}"
)

print(
    f"Initial max occupancy:   "
    f"{initial_max_occupancy}"
)

print(
    f"Timestep:                "
    f"{dt * 1e6:.2f} us"
)


print()
print("RESULTS")
print("------------------------------------------")

print(
    f"Allocation time:         "
    f"{allocation_time:.3f} s"
)

print(
    f"Packed particle buffers: "
    f"{raw_particle_bytes / 1024**2:.2f} MiB"
)

print(
    f"Neighbor table:          "
    f"{neighbor_bytes / 1024**2:.2f} MiB"
)

print(
    f"CuPy memory used:        "
    f"{pool.used_bytes() / 1024**2:.2f} MiB"
)

print(
    f"CuPy memory reserved:    "
    f"{pool.total_bytes() / 1024**2:.2f} MiB"
)

print()

print(
    f"Diffusion/update:        "
    f"{diffusion_ms / args.steps:.4f} ms"
)

print(
    f"Local scan/update:       "
    f"{scan_ms / args.steps:.4f} ms"
)

print()

print(
    f"Final max occupancy:     "
    f"{max_occupancy_final}"
)

print(
    f"Overflow events:         "
    f"{overflow_count:,}"
)

print(
    f"Final stored molecules:  "
    f"{final_molecules:,}"
)

print(
    f"Scan checksum:           "
    f"{int(cp.sum(scan_result)):,}"
)
