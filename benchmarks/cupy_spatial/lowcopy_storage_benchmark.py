import argparse
import time
import numpy as np
import cupy as cp


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["dense", "sparse"],
    required=True
)

parser.add_argument(
    "--spacing-nm",
    type=float,
    default=50.0
)

parser.add_argument(
    "--species",
    type=int,
    required=True
)

parser.add_argument(
    "--molecules",
    type=int,
    default=250000
)

parser.add_argument(
    "--steps",
    type=int,
    default=100
)

parser.add_argument(
    "--D",
    type=float,
    default=1.0,
    help="Reference diffusion coefficient, um^2/s"
)

args = parser.parse_args()


# ============================================================
# Geometry
#
# All geometry derives from voxel size.
# ============================================================

spacing_um = args.spacing_nm / 1000.0

CELL_LENGTH = 2.4
CELL_DIAMETER = 0.9
RADIUS = CELL_DIAMETER / 2.0

HALF_CYLINDER = (
    CELL_LENGTH - CELL_DIAMETER
) / 2.0

NX = int(
    np.ceil(
        (CELL_LENGTH + 2 * spacing_um)
        / spacing_um
    )
)

NY = int(
    np.ceil(
        (CELL_DIAMETER + 2 * spacing_um)
        / spacing_um
    )
)

NZ = NY

x = (
    np.arange(NX) - (NX - 1) / 2
) * spacing_um

y = (
    np.arange(NY) - (NY - 1) / 2
) * spacing_um

z = (
    np.arange(NZ) - (NZ - 1) / 2
) * spacing_um

X, Y, Z = np.meshgrid(
    x, y, z,
    indexing="ij"
)

axial_excess = np.maximum(
    np.abs(X) - HALF_CYLINDER,
    0.0
)

mask_cpu = (
    axial_excess**2
    + Y**2
    + Z**2
) <= RADIUS**2

mask_flat = mask_cpu.reshape(-1)

NVOX = NX * NY * NZ
CELL_VOXELS = int(mask_flat.sum())


# ============================================================
# Diffusion timestep
#
# Same rule as previous benchmark:
#
# total movement probability = 0.5
# ============================================================

q = 0.5 / 6.0

dt = (
    q
    * spacing_um**2
    / args.D
)


# ============================================================
# Neighbor table
#
# neighbor[voxel, direction]
#
# If movement would leave the cell, destination = original
# voxel -> reflecting boundary.
# ============================================================

neighbors_cpu = np.zeros(
    (NVOX, 6),
    dtype=np.int32
)

plane = NY * NZ

for voxel in range(NVOX):

    x_i = voxel // plane

    rem = voxel - x_i * plane

    y_i = rem // NZ
    z_i = rem - y_i * NZ

    candidates = [
        voxel + plane if x_i + 1 < NX else voxel,
        voxel - plane if x_i > 0 else voxel,
        voxel + NZ if y_i + 1 < NY else voxel,
        voxel - NZ if y_i > 0 else voxel,
        voxel + 1 if z_i + 1 < NZ else voxel,
        voxel - 1 if z_i > 0 else voxel,
    ]

    for d, candidate in enumerate(candidates):

        if (
            mask_flat[voxel]
            and mask_flat[candidate]
        ):
            neighbors_cpu[voxel, d] = candidate
        else:
            neighbors_cpu[voxel, d] = voxel


neighbors = cp.asarray(
    neighbors_cpu.reshape(-1)
)

cell_indices = cp.asarray(
    np.flatnonzero(mask_flat),
    dtype=cp.int32
)


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
    unsigned long long x = mix64(key);

    return
        ((x >> 40) & 0xFFFFFFULL)
        * (1.0f / 16777216.0f);
}


// ============================================================
// DENSE DIFFUSION
//
// One count exists for every species x voxel combination.
// ============================================================

extern "C" __global__
void dense_diffuse(
    const unsigned int* input,
    unsigned int* output,
    const int* neighbors,
    long long nvox,
    long long total,
    float q,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    unsigned int n = input[idx];

    if (n == 0)
        return;

    long long species =
        idx / nvox;

    int voxel =
        (int)(idx % nvox);

    for (
        unsigned int molecule = 0;
        molecule < n;
        molecule++
    )
    {
        unsigned long long key =
            ((unsigned long long)idx
             * 0x9e3779b97f4a7c15ULL)
            ^
            ((unsigned long long)step
             * 0xbf58476d1ce4e5b9ULL)
            ^
            molecule;

        float r = random01(key);

        int destination = voxel;

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

        atomicAdd(
            &output[
                species * nvox
                + destination
            ],
            1U
        );
    }
}


// ============================================================
// SPARSE DIFFUSION
//
// There is one entry per actual molecule.
//
// No empty species x voxel locations are visited.
// ============================================================

extern "C" __global__
void sparse_diffuse(
    int* voxels,
    const int* neighbors,
    long long molecules,
    float q,
    unsigned long long step)
{
    long long i =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (i >= molecules)
        return;

    int voxel = voxels[i];

    unsigned long long key =
        ((unsigned long long)i
         * 0x9e3779b97f4a7c15ULL)
        ^
        ((unsigned long long)step
         * 0xbf58476d1ce4e5b9ULL);

    float r = random01(key);

    if (r < 6.0f * q)
    {
        int direction =
            (int)(r / q);

        if (direction > 5)
            direction = 5;

        voxels[i] =
            neighbors[
                voxel * 6
                + direction
            ];
    }
}


// ============================================================
// DENSE FIRST-ORDER DECAY
// ============================================================

extern "C" __global__
void dense_decay(
    unsigned int* counts,
    const float* rates,
    long long nvox,
    long long total,
    float dt,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    unsigned int n =
        counts[idx];

    if (n == 0)
        return;

    long long species =
        idx / nvox;

    float p =
        1.0f
        - expf(
            -rates[species] * dt
        );

    unsigned int survivors = 0;

    for (
        unsigned int molecule = 0;
        molecule < n;
        molecule++
    )
    {
        unsigned long long key =
            ((unsigned long long)idx
             * 0xd6e8feb86659fd93ULL)
            ^
            ((unsigned long long)step
             * 0xa5a35625aae3e29dULL)
            ^
            molecule;

        if (random01(key) >= p)
            survivors++;
    }

    counts[idx] = survivors;
}


// ============================================================
// SPARSE DECAY MARKING
//
// Marks molecules dead/alive.
// Actual compaction can be done less frequently.
// ============================================================

extern "C" __global__
void sparse_decay_mark(
    const unsigned short* species,
    unsigned char* alive,
    const float* rates,
    long long molecules,
    float dt,
    unsigned long long step)
{
    long long i =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (i >= molecules)
        return;

    if (!alive[i])
        return;

    unsigned short s =
        species[i];

    float p =
        1.0f
        - expf(
            -rates[s] * dt
        );

    unsigned long long key =
        ((unsigned long long)i
         * 0xd6e8feb86659fd93ULL)
        ^
        ((unsigned long long)step
         * 0xa5a35625aae3e29dULL);

    if (random01(key) < p)
        alive[i] = 0;
}

'''


module = cp.RawModule(
    code=CUDA_SRC
)

dense_diffuse = module.get_function(
    "dense_diffuse"
)

sparse_diffuse = module.get_function(
    "sparse_diffuse"
)

dense_decay = module.get_function(
    "dense_decay"
)

sparse_decay_mark = module.get_function(
    "sparse_decay_mark"
)


THREADS = 256


def blocks(total):

    return (
        int(
            (total + THREADS - 1)
            // THREADS
        ),
    )


def time_gpu(function):

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
# Shared initialization
# ============================================================

species_initial = cp.random.randint(
    0,
    args.species,
    args.molecules,
    dtype=cp.int32
)

voxel_initial = cell_indices[
    cp.random.randint(
        0,
        CELL_VOXELS,
        args.molecules
    )
]

rates = cp.linspace(
    0.01,
    0.20,
    args.species,
    dtype=cp.float32
)

cp.cuda.Stream.null.synchronize()


print()
print("CONFIGURATION")
print("--------------------------------------")
print(f"Mode:                 {args.mode}")
print(f"Voxel size:           {args.spacing_nm:.1f} nm")
print(f"Grid:                 {NX} x {NY} x {NZ}")
print(f"Bounding voxels:      {NVOX:,}")
print(f"Cell voxels:          {CELL_VOXELS:,}")
print(f"Species:              {args.species:,}")
print(f"Molecules:            {args.molecules:,}")
print(f"Timestep:             {dt*1e6:.2f} us")
print(f"Benchmark steps:      {args.steps:,}")


# ============================================================
# DENSE MODE
# ============================================================

if args.mode == "dense":

    total_values = (
        args.species
        * NVOX
    )

    alloc_start = time.perf_counter()

    counts_a = cp.zeros(
        total_values,
        dtype=cp.uint32
    )

    counts_b = cp.zeros_like(
        counts_a
    )

    initial_keys = (
        species_initial.astype(cp.int64)
        * NVOX
        + voxel_initial.astype(cp.int64)
    )

    cp.add.at(
        counts_a,
        initial_keys,
        1
    )

    cp.cuda.Stream.null.synchronize()

    alloc_time = (
        time.perf_counter()
        - alloc_start
    )

    # Warmup
    counts_b.fill(0)

    dense_diffuse(
        blocks(total_values),
        (THREADS,),
        (
            counts_a,
            counts_b,
            neighbors,
            np.int64(NVOX),
            np.int64(total_values),
            np.float32(q),
            np.uint64(0)
        )
    )

    cp.cuda.Stream.null.synchronize()

    # Reinitialize after warmup
    counts_a.fill(0)

    cp.add.at(
        counts_a,
        initial_keys,
        1
    )

    cp.cuda.Stream.null.synchronize()

    def run_diffusion():

        nonlocal_dummy = None

        global counts_a, counts_b

        for step in range(args.steps):

            counts_b.fill(0)

            dense_diffuse(
                blocks(total_values),
                (THREADS,),
                (
                    counts_a,
                    counts_b,
                    neighbors,
                    np.int64(NVOX),
                    np.int64(total_values),
                    np.float32(q),
                    np.uint64(step)
                )
            )

            counts_a, counts_b = (
                counts_b,
                counts_a
            )

    diffusion_ms = time_gpu(
        run_diffusion
    )

    def run_decay():

        for step in range(args.steps):

            dense_decay(
                blocks(total_values),
                (THREADS,),
                (
                    counts_a,
                    rates,
                    np.int64(NVOX),
                    np.int64(total_values),
                    np.float32(dt),
                    np.uint64(step)
                )
            )

    decay_ms = time_gpu(
        run_decay
    )

    raw_state_bytes = (
        counts_a.nbytes
        + counts_b.nbytes
    )

    molecule_count = int(
        cp.sum(counts_a)
    )

    pool = cp.get_default_memory_pool()

    print()
    print("RESULTS")
    print("--------------------------------------")

    print(
        f"Allocation time:       "
        f"{alloc_time:.3f} s"
    )

    print(
        f"Raw count buffers:     "
        f"{raw_state_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"CuPy memory used:      "
        f"{pool.used_bytes()/1024**2:.2f} MiB"
    )

    print(
        f"CuPy memory reserved:  "
        f"{pool.total_bytes()/1024**2:.2f} MiB"
    )

    print()

    print(
        f"Diffusion/update:      "
        f"{diffusion_ms/args.steps:.4f} ms"
    )

    print(
        f"Decay/update:          "
        f"{decay_ms/args.steps:.4f} ms"
    )

    print()

    print(
        f"Final molecules:       "
        f"{molecule_count:,}"
    )


# ============================================================
# SPARSE MODE
# ============================================================

else:

    if args.species > 65535:

        raise ValueError(
            "This benchmark uses uint16 species IDs. "
            "Increase storage type if >65535 species."
        )

    alloc_start = time.perf_counter()

    species_sparse = (
        species_initial
        .astype(cp.uint16)
        .copy()
    )

    voxel_sparse = (
        voxel_initial
        .astype(cp.int32)
        .copy()
    )

    alive = cp.ones(
        args.molecules,
        dtype=cp.uint8
    )

    cp.cuda.Stream.null.synchronize()

    alloc_time = (
        time.perf_counter()
        - alloc_start
    )

    # Warmup
    sparse_diffuse(
        blocks(args.molecules),
        (THREADS,),
        (
            voxel_sparse,
            neighbors,
            np.int64(args.molecules),
            np.float32(q),
            np.uint64(0)
        )
    )

    cp.cuda.Stream.null.synchronize()

    voxel_sparse = (
        voxel_initial
        .astype(cp.int32)
        .copy()
    )

    # --------------------------------------------------------
    # Sparse diffusion
    # --------------------------------------------------------

    def run_sparse_diffusion():

        for step in range(args.steps):

            sparse_diffuse(
                blocks(args.molecules),
                (THREADS,),
                (
                    voxel_sparse,
                    neighbors,
                    np.int64(args.molecules),
                    np.float32(q),
                    np.uint64(step)
                )
            )

    diffusion_ms = time_gpu(
        run_sparse_diffusion
    )

    # --------------------------------------------------------
    # Sparse degradation marking
    # --------------------------------------------------------

    def run_sparse_decay():

        for step in range(args.steps):

            sparse_decay_mark(
                blocks(args.molecules),
                (THREADS,),
                (
                    species_sparse,
                    alive,
                    rates,
                    np.int64(args.molecules),
                    np.float32(dt),
                    np.uint64(step)
                )
            )

    decay_ms = time_gpu(
        run_sparse_decay
    )

    # --------------------------------------------------------
    # Build sparse occupancy index.
    #
    # For reactions we often need:
    #
    #   species X in voxel Y = how many?
    #
    # Generate key:
    #
    #   key = species * NVOX + voxel
    #
    # Sort identical keys together and count them.
    # --------------------------------------------------------

    def build_occupancy():

        active_species = (
            species_sparse[
                alive.astype(cp.bool_)
            ]
        )

        active_voxels = (
            voxel_sparse[
                alive.astype(cp.bool_)
            ]
        )

        keys = (
            active_species.astype(cp.uint64)
            * np.uint64(NVOX)
            + active_voxels.astype(cp.uint64)
        )

        sorted_keys = cp.sort(keys)

        if sorted_keys.size == 0:

            unique_keys = sorted_keys
            occupied_counts = cp.empty(
                0,
                dtype=cp.int32
            )

        else:

            starts_mask = cp.empty(
                sorted_keys.size,
                dtype=cp.bool_
            )

            starts_mask[0] = True

            starts_mask[1:] = (
                sorted_keys[1:]
                != sorted_keys[:-1]
            )

            starts = cp.flatnonzero(
                starts_mask
            )

            unique_keys = (
                sorted_keys[starts]
            )

            ends = cp.concatenate(
                (
                    starts[1:],
                    cp.array(
                        [sorted_keys.size],
                        dtype=starts.dtype
                    )
                )
            )

            occupied_counts = (
                ends - starts
            )

        # Force execution before timing ends
        if unique_keys.size:
            _ = int(
                cp.sum(occupied_counts)
            )

        return (
            unique_keys,
            occupied_counts
        )

    occupancy_ms = time_gpu(
        build_occupancy
    )

    unique_keys, occupied_counts = (
        build_occupancy()
    )

    cp.cuda.Stream.null.synchronize()

    raw_state_bytes = (
        species_sparse.nbytes
        + voxel_sparse.nbytes
        + alive.nbytes
    )

    alive_count = int(
        cp.sum(alive)
    )

    pool = cp.get_default_memory_pool()

    print()
    print("RESULTS")
    print("--------------------------------------")

    print(
        f"Allocation time:       "
        f"{alloc_time:.3f} s"
    )

    print(
        f"Raw sparse state:      "
        f"{raw_state_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"CuPy memory used:      "
        f"{pool.used_bytes()/1024**2:.2f} MiB"
    )

    print(
        f"CuPy memory reserved:  "
        f"{pool.total_bytes()/1024**2:.2f} MiB"
    )

    print()

    print(
        f"Diffusion/update:      "
        f"{diffusion_ms/args.steps:.4f} ms"
    )

    print(
        f"Decay marking/update:  "
        f"{decay_ms/args.steps:.4f} ms"
    )

    print(
        f"Occupancy-index build: "
        f"{occupancy_ms:.4f} ms"
    )

    print()

    print(
        f"Occupied species/voxel pairs: "
        f"{unique_keys.size:,}"
    )

    print(
        f"Living molecules:      "
        f"{alive_count:,}"
    )
