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
parser.add_argument("--molecules", type=int, default=500000)

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

parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--D", type=float, default=1.0)

args = parser.parse_args()


# ============================================================
# Geometry
#
# Again: everything derives from voxel size.
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

active_flat = np.flatnonzero(mask_flat)

CELL_VOXELS = active_flat.size


# ============================================================
# Compact voxel map
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
# Neighbor table
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

    for d, candidate in enumerate(candidates):

        candidate_active = flat_to_active[candidate]

        if candidate_active >= 0:
            neighbors_cpu[
                active_id,
                d
            ] = candidate_active
        else:
            neighbors_cpu[
                active_id,
                d
            ] = active_id


neighbors = cp.asarray(
    neighbors_cpu.reshape(-1)
)


# ============================================================
# Diffusion timestep
# ============================================================

q = 0.5 / 6.0

dt = (
    q
    * spacing_um**2
    / args.D
)


# ============================================================
# Initialize molecules
# ============================================================

if args.species > 65534:

    raise ValueError(
        "uint16 species IDs support <=65,534 species."
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


# ============================================================
# Pack into primary + spill
# ============================================================

order = np.argsort(
    initial_voxels,
    kind="stable"
)

vox_sorted = initial_voxels[order]
species_sorted = initial_species[order]

total_occupancy = np.bincount(
    vox_sorted,
    minlength=CELL_VOXELS
).astype(np.uint32)

initial_max_occupancy = int(
    total_occupancy.max()
)

TOTAL_CAPACITY = (
    args.primary_slots
    + args.spill_slots
)

if initial_max_occupancy > TOTAL_CAPACITY:

    raise RuntimeError(
        f"Initial occupancy "
        f"{initial_max_occupancy} exceeds "
        f"primary+spill capacity "
        f"{TOTAL_CAPACITY}."
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


for voxel in range(CELL_VOXELS):

    species_here = species_sorted[
        vox_sorted == voxel
    ]

    n = len(species_here)

    n_primary = min(
        n,
        args.primary_slots
    )

    n_spill = max(
        0,
        n - args.primary_slots
    )

    if n_primary:
        primary_cpu[
            voxel,
            :n_primary
        ] = species_here[
            :n_primary
        ]

    if n_spill:
        spill_cpu[
            voxel,
            :n_spill
        ] = species_here[
            n_primary:
        ]

    primary_occ_cpu[voxel] = n_primary
    spill_occ_cpu[voxel] = n_spill


# ============================================================
# CUDA kernels
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
    unsigned long long x =
        mix64(key);

    return (
        ((x >> 40) & 0xFFFFFFULL)
        * (1.0f / 16777216.0f)
    );
}


// ============================================================
// Safely reserve a slot without allowing occupancy
// to exceed its capacity.
// ============================================================

__device__ __forceinline__
bool reserve_slot(
    unsigned int* occupancy,
    int voxel,
    unsigned int capacity,
    unsigned int* slot_out)
{
    while (true)
    {
        unsigned int old =
            atomicAdd(
                &occupancy[voxel],
                0U
            );

        if (old >= capacity)
            return false;

        unsigned int previous =
            atomicCAS(
                &occupancy[voxel],
                old,
                old + 1
            );

        if (previous == old)
        {
            *slot_out = old;
            return true;
        }
    }
}


// ============================================================
// Place molecule into PRIMARY first, SPILL second.
// ============================================================

__device__ __forceinline__
void place_particle(
    unsigned short species,

    int voxel,

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

    unsigned int* spill_writes,
    unsigned int* hard_overflow,

    int primary_slots,
    int spill_slots)
{
    unsigned int slot;

    if (
        reserve_slot(
            primary_occ,
            voxel,
            primary_slots,
            &slot
        )
    )
    {
        primary[
            (long long)voxel
            * primary_slots
            + slot
        ] = species;

        return;
    }


    if (
        reserve_slot(
            spill_occ,
            voxel,
            spill_slots,
            &slot
        )
    )
    {
        spill[
            (long long)voxel
            * spill_slots
            + slot
        ] = species;

        atomicAdd(
            spill_writes,
            1U
        );

        return;
    }


    // Both primary and spill are full.
    atomicAdd(
        hard_overflow,
        1U
    );
}


// ============================================================
// Diffusion from PRIMARY input
// ============================================================

extern "C" __global__
void diffuse_primary(
    const unsigned short* primary_in,
    const unsigned int* primary_occ_in,

    unsigned short* primary_out,
    unsigned int* primary_occ_out,

    unsigned short* spill_out,
    unsigned int* spill_occ_out,

    const int* neighbors,

    unsigned int* spill_writes,
    unsigned int* hard_overflow,

    int primary_slots,
    int spill_slots,

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
        * (long long)primary_slots;

    if (idx >= total)
        return;

    int voxel =
        idx / primary_slots;

    int slot =
        idx % primary_slots;

    if (
        slot >= primary_occ_in[voxel]
    )
        return;


    unsigned short species =
        primary_in[idx];

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


    place_particle(
        species,
        destination,

        primary_out,
        primary_occ_out,

        spill_out,
        spill_occ_out,

        spill_writes,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}


// ============================================================
// Diffusion from SPILL input
//
// Important: spill particles still diffuse normally.
// They are not permanently stuck in the spill area.
// ============================================================

extern "C" __global__
void diffuse_spill(
    const unsigned short* spill_in,
    const unsigned int* spill_occ_in,

    unsigned short* primary_out,
    unsigned int* primary_occ_out,

    unsigned short* spill_out,
    unsigned int* spill_occ_out,

    const int* neighbors,

    unsigned int* spill_writes,
    unsigned int* hard_overflow,

    int primary_slots,
    int spill_slots,

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
        * (long long)spill_slots;

    if (idx >= total)
        return;

    int voxel =
        idx / spill_slots;

    int slot =
        idx % spill_slots;

    if (
        slot >= spill_occ_in[voxel]
    )
        return;


    unsigned short species =
        spill_in[idx];

    if (species == 0)
        return;


    unsigned long long key =
        ((unsigned long long)idx
         * 0xd6e8feb86659fd93ULL)
        ^
        ((unsigned long long)step
         * 0xa5a35625aae3e29dULL);

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


    // Try primary again first.
    //
    // A molecule that was spilled last step can move back
    // into normal primary storage this step.
    place_particle(
        species,
        destination,

        primary_out,
        primary_occ_out,

        spill_out,
        spill_occ_out,

        spill_writes,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}


// ============================================================
// Local voxel scan INCLUDING spill molecules.
// ============================================================

extern "C" __global__
void local_scan(
    const unsigned short* primary,
    const unsigned int* primary_occ,

    const unsigned short* spill,
    const unsigned int* spill_occ,

    unsigned int* totals,

    int primary_slots,
    int spill_slots,

    long long nvox)
{
    long long voxel =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (voxel >= nvox)
        return;


    unsigned int count = 0;

    unsigned int np =
        primary_occ[voxel];

    unsigned int ns =
        spill_occ[voxel];


    for (
        unsigned int i = 0;
        i < np;
        i++
    )
    {
        if (
            primary[
                voxel
                * (long long)primary_slots
                + i
            ] != 0
        )
        {
            count++;
        }
    }


    for (
        unsigned int i = 0;
        i < ns;
        i++
    )
    {
        if (
            spill[
                voxel
                * (long long)spill_slots
                + i
            ] != 0
        )
        {
            count++;
        }
    }


    totals[voxel] = count;
}

'''


module = cp.RawModule(
    code=CUDA_SRC
)

diffuse_primary = module.get_function(
    "diffuse_primary"
)

diffuse_spill = module.get_function(
    "diffuse_spill"
)

local_scan = module.get_function(
    "local_scan"
)


# ============================================================
# GPU state
# ============================================================

primary_a = cp.asarray(
    primary_cpu.reshape(-1)
)

primary_b = cp.zeros_like(
    primary_a
)

spill_a = cp.asarray(
    spill_cpu.reshape(-1)
)

spill_b = cp.zeros_like(
    spill_a
)

primary_occ_a = cp.asarray(
    primary_occ_cpu
)

primary_occ_b = cp.zeros_like(
    primary_occ_a
)

spill_occ_a = cp.asarray(
    spill_occ_cpu
)

spill_occ_b = cp.zeros_like(
    spill_occ_a
)

spill_writes = cp.zeros(
    1,
    dtype=cp.uint32
)

hard_overflow = cp.zeros(
    1,
    dtype=cp.uint32
)

scan_totals = cp.zeros(
    CELL_VOXELS,
    dtype=cp.uint32
)


THREADS = 256


PRIMARY_TOTAL = (
    CELL_VOXELS
    * args.primary_slots
)

SPILL_TOTAL = (
    CELL_VOXELS
    * args.spill_slots
)


def blocks(total):

    return (
        int(
            (total + THREADS - 1)
            // THREADS
        ),
    )


def gpu_time(fn):

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    fn()

    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(
        start,
        end
    )


# ============================================================
# Warmup
# ============================================================

local_scan(
    blocks(CELL_VOXELS),
    (THREADS,),
    (
        primary_a,
        primary_occ_a,

        spill_a,
        spill_occ_a,

        scan_totals,

        np.int32(args.primary_slots),
        np.int32(args.spill_slots),

        np.int64(CELL_VOXELS)
    )
)

cp.cuda.Stream.null.synchronize()


# ============================================================
# Diffusion benchmark
# ============================================================

def run_diffusion():

    global primary_a
    global primary_b
    global spill_a
    global spill_b

    global primary_occ_a
    global primary_occ_b
    global spill_occ_a
    global spill_occ_b

    for step in range(args.steps):

        primary_occ_b.fill(0)
        spill_occ_b.fill(0)

        # Primary molecules move.
        diffuse_primary(
            blocks(PRIMARY_TOTAL),
            (THREADS,),
            (
                primary_a,
                primary_occ_a,

                primary_b,
                primary_occ_b,

                spill_b,
                spill_occ_b,

                neighbors,

                spill_writes,
                hard_overflow,

                np.int32(
                    args.primary_slots
                ),

                np.int32(
                    args.spill_slots
                ),

                np.int64(
                    CELL_VOXELS
                ),

                np.float32(q),

                np.uint64(step)
            )
        )


        # Spill molecules also move.
        diffuse_spill(
            blocks(SPILL_TOTAL),
            (THREADS,),
            (
                spill_a,
                spill_occ_a,

                primary_b,
                primary_occ_b,

                spill_b,
                spill_occ_b,

                neighbors,

                spill_writes,
                hard_overflow,

                np.int32(
                    args.primary_slots
                ),

                np.int32(
                    args.spill_slots
                ),

                np.int64(
                    CELL_VOXELS
                ),

                np.float32(q),

                np.uint64(step)
            )
        )


        primary_a, primary_b = (
            primary_b,
            primary_a
        )

        spill_a, spill_b = (
            spill_b,
            spill_a
        )

        primary_occ_a, primary_occ_b = (
            primary_occ_b,
            primary_occ_a
        )

        spill_occ_a, spill_occ_b = (
            spill_occ_b,
            spill_occ_a
        )


diffusion_ms = gpu_time(
    run_diffusion
)


# ============================================================
# Scan benchmark
# ============================================================

def run_scan():

    for _ in range(args.steps):

        local_scan(
            blocks(CELL_VOXELS),
            (THREADS,),
            (
                primary_a,
                primary_occ_a,

                spill_a,
                spill_occ_a,

                scan_totals,

                np.int32(
                    args.primary_slots
                ),

                np.int32(
                    args.spill_slots
                ),

                np.int64(
                    CELL_VOXELS
                )
            )
        )


scan_ms = gpu_time(
    run_scan
)


# ============================================================
# Results
# ============================================================

final_primary = int(
    cp.sum(primary_occ_a)
)

final_spill = int(
    cp.sum(spill_occ_a)
)

final_total = (
    final_primary
    + final_spill
)

max_primary = int(
    cp.max(primary_occ_a)
)

max_spill = int(
    cp.max(spill_occ_a)
)

spill_write_count = int(
    spill_writes[0]
)

hard_overflow_count = int(
    hard_overflow[0]
)

scan_total = int(
    cp.sum(scan_totals)
)


particle_bytes = (
    primary_a.nbytes
    + primary_b.nbytes
    + spill_a.nbytes
    + spill_b.nbytes
)

pool = cp.get_default_memory_pool()


print()
print("CONFIGURATION")
print("--------------------------------------------")

print(
    f"Voxel size:                "
    f"{args.spacing_nm:.1f} nm"
)

print(
    f"Cell voxels:               "
    f"{CELL_VOXELS:,}"
)

print(
    f"Species universe:          "
    f"{args.species:,}"
)

print(
    f"Initial molecules:         "
    f"{args.molecules:,}"
)

print(
    f"Primary slots / voxel:     "
    f"{args.primary_slots}"
)

print(
    f"Spill slots / voxel:       "
    f"{args.spill_slots}"
)

print(
    f"Total capacity / voxel:    "
    f"{TOTAL_CAPACITY}"
)

print(
    f"Initial max occupancy:     "
    f"{initial_max_occupancy}"
)

print(
    f"Timestep:                  "
    f"{dt*1e6:.2f} us"
)


print()
print("MEMORY")
print("--------------------------------------------")

print(
    f"Particle buffers:          "
    f"{particle_bytes/1024**2:.2f} MiB"
)

print(
    f"CuPy memory used:          "
    f"{pool.used_bytes()/1024**2:.2f} MiB"
)

print(
    f"CuPy memory reserved:      "
    f"{pool.total_bytes()/1024**2:.2f} MiB"
)


print()
print("PERFORMANCE")
print("--------------------------------------------")

print(
    f"Diffusion / update:        "
    f"{diffusion_ms/args.steps:.4f} ms"
)

print(
    f"Local scan / update:       "
    f"{scan_ms/args.steps:.4f} ms"
)


print()
print("FINAL STATE")
print("--------------------------------------------")

print(
    f"Primary molecules:         "
    f"{final_primary:,}"
)

print(
    f"Spill molecules:           "
    f"{final_spill:,}"
)

print(
    f"Total stored molecules:    "
    f"{final_total:,}"
)

print(
    f"Final max primary occ:     "
    f"{max_primary}"
)

print(
    f"Final max spill occ:       "
    f"{max_spill}"
)

print(
    f"Cumulative spill writes:   "
    f"{spill_write_count:,}"
)

print(
    f"HARD overflow events:      "
    f"{hard_overflow_count:,}"
)

print(
    f"Scan molecule count:       "
    f"{scan_total:,}"
)


print()
print("CONSERVATION")
print("--------------------------------------------")

difference = (
    final_total
    - args.molecules
)

print(
    f"Initial:                   "
    f"{args.molecules:,}"
)

print(
    f"Final:                     "
    f"{final_total:,}"
)

print(
    f"Difference:                "
    f"{difference:+,}"
)

if (
    difference == 0
    and hard_overflow_count == 0
):
    print("PASS: all molecules conserved")

else:
    print("FAIL: molecules were lost")
