import argparse
import time
import numpy as np
import cupy as cp


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--spacing-nm", type=float, required=True)

parser.add_argument(
    "--count-species",
    type=int,
    default=512,
    help="Low-copy species stored as integer molecule counts"
)

parser.add_argument(
    "--field-species",
    type=int,
    default=2048,
    help="Abundant species stored as float concentration fields"
)

parser.add_argument(
    "--particles",
    type=int,
    default=100000,
    help="Total molecules distributed among count species"
)

parser.add_argument(
    "--reaction-pairs",
    type=int,
    default=128
)

parser.add_argument(
    "--steps",
    type=int,
    default=50
)

parser.add_argument(
    "--D",
    type=float,
    default=1.0,
    help="Reference diffusion coefficient in um^2/s"
)

args = parser.parse_args()


# ============================================================
# Geometry
#
# E. coli-like spherocylinder:
#
# total length = 2.4 um
# diameter     = 0.9 um
#
# volume ~1.3 um^3
# ============================================================

spacing_um = args.spacing_nm / 1000.0

CELL_LENGTH = 2.4
CELL_DIAMETER = 0.9
RADIUS = CELL_DIAMETER / 2.0

# Straight cylindrical section between hemispherical caps
HALF_CYLINDER = (CELL_LENGTH - CELL_DIAMETER) / 2.0

# One-voxel padding around the cell prevents periodic wraparound
NX = int(np.ceil((CELL_LENGTH + 2 * spacing_um) / spacing_um))
NY = int(np.ceil((CELL_DIAMETER + 2 * spacing_um) / spacing_um))
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

# Distance from central line segment of the spherocylinder
axial_excess = np.maximum(
    np.abs(X) - HALF_CYLINDER,
    0.0
)

mask_cpu = (
    axial_excess**2
    + Y**2
    + Z**2
) <= RADIUS**2

mask = cp.asarray(
    mask_cpu.reshape(-1).astype(np.uint8)
)

NVOX = NX * NY * NZ
CELL_VOXELS = int(mask_cpu.sum())


# ============================================================
# Diffusion timestep
#
# q = D * dt / dx^2
#
# Six neighbors.
# Choose total hop probability = 0.5:
#
# 6q = 0.5
# ============================================================

q = 0.5 / 6.0

dt = q * spacing_um**2 / args.D

print()
print("GEOMETRY")
print("--------------------------------------------")
print(f"Voxel spacing:       {args.spacing_nm:.1f} nm")
print(f"Grid:                {NX} x {NY} x {NZ}")
print(f"Bounding voxels:     {NVOX:,}")
print(f"Cell voxels:         {CELL_VOXELS:,}")
print(f"Cell volume:         {CELL_VOXELS * spacing_um**3:.3f} um^3")
print(f"Reference D:         {args.D:.2f} um^2/s")
print(f"Timestep:            {dt * 1e6:.2f} us")

print()
print("STATE")
print("--------------------------------------------")
print(f"Count species:       {args.count_species:,}")
print(f"Field species:       {args.field_species:,}")
print(f"Explicit molecules:  {args.particles:,}")


# ============================================================
# CUDA kernels
# ============================================================

CUDA_SRC = r'''

__device__ __forceinline__
unsigned long long mix64(unsigned long long x)
{
    x += 0x9e3779b97f4a7c15ULL;

    x = (x ^ (x >> 30))
        * 0xbf58476d1ce4e5b9ULL;

    x = (x ^ (x >> 27))
        * 0x94d049bb133111ebULL;

    return x ^ (x >> 31);
}


__device__ __forceinline__
float random01(unsigned long long key)
{
    unsigned long long x = mix64(key);

    return
        ((x >> 40) & 0xFFFFFFULL)
        * (1.0f / 16777216.0f);
}


// ============================================================
// INTEGER MOLECULE DIFFUSION
//
// Each molecule independently either:
//
// stay
// +x
// -x
// +y
// -y
// +z
// -z
//
// Molecules hitting the cell boundary stay in place.
// ============================================================

extern "C" __global__
void diffuse_counts(
    const unsigned int* input,
    unsigned int* output,
    const unsigned char* mask,
    int nx,
    int ny,
    int nz,
    long long nvox,
    long long total,
    float q,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    unsigned int n = input[idx];

    if (n == 0)
        return;

    long long voxel = idx % nvox;
    long long species = idx / nvox;

    long long plane =
        (long long)ny * nz;

    int x = voxel / plane;

    long long remainder =
        voxel - (long long)x * plane;

    int y = remainder / nz;
    int z = remainder - (long long)y * nz;

    for (unsigned int m = 0; m < n; m++)
    {
        unsigned long long key =
            ((unsigned long long)idx * 0x9e3779b97f4a7c15ULL)
            ^ ((unsigned long long)step * 0xbf58476d1ce4e5b9ULL)
            ^ ((unsigned long long)m * 0x94d049bb133111ebULL);

        float r = random01(key);

        long long destination = voxel;

        if (r < 6.0f * q)
        {
            int direction =
                (int)(r / q);

            long long candidate = voxel;

            if (direction == 0 && x + 1 < nx)
                candidate = voxel + plane;

            else if (direction == 1 && x > 0)
                candidate = voxel - plane;

            else if (direction == 2 && y + 1 < ny)
                candidate = voxel + nz;

            else if (direction == 3 && y > 0)
                candidate = voxel - nz;

            else if (direction == 4 && z + 1 < nz)
                candidate = voxel + 1;

            else if (direction == 5 && z > 0)
                candidate = voxel - 1;

            if (mask[candidate])
                destination = candidate;
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
// FLOAT CONCENTRATION DIFFUSION
//
// Standard 3-D finite-difference diffusion with reflecting
// cell boundaries.
// ============================================================

extern "C" __global__
void diffuse_fields(
    const float* input,
    float* output,
    const unsigned char* mask,
    int nx,
    int ny,
    int nz,
    long long nvox,
    long long total,
    float alpha)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    long long voxel = idx % nvox;

    if (!mask[voxel])
    {
        output[idx] = 0.0f;
        return;
    }

    long long plane =
        (long long)ny * nz;

    int x = voxel / plane;

    long long remainder =
        voxel - (long long)x * plane;

    int y = remainder / nz;
    int z = remainder - (long long)y * nz;

    float center = input[idx];

    float lap = 0.0f;

    long long n;

    if (x + 1 < nx)
    {
        n = voxel + plane;

        if (mask[n])
            lap += input[idx + plane] - center;
    }

    if (x > 0)
    {
        n = voxel - plane;

        if (mask[n])
            lap += input[idx - plane] - center;
    }

    if (y + 1 < ny)
    {
        n = voxel + nz;

        if (mask[n])
            lap += input[idx + nz] - center;
    }

    if (y > 0)
    {
        n = voxel - nz;

        if (mask[n])
            lap += input[idx - nz] - center;
    }

    if (z + 1 < nz)
    {
        n = voxel + 1;

        if (mask[n])
            lap += input[idx + 1] - center;
    }

    if (z > 0)
    {
        n = voxel - 1;

        if (mask[n])
            lap += input[idx - 1] - center;
    }

    output[idx] =
        center + alpha * lap;
}


// ============================================================
// GENERIC FIRST-ORDER COUNT REACTION
//
// Example:
//      mRNA -> nothing
//
// Every count species has its own rate.
// ============================================================

extern "C" __global__
void count_decay(
    unsigned int* counts,
    const float* rates,
    long long nvox,
    long long total,
    float dt,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    unsigned int n = counts[idx];

    if (n == 0)
        return;

    long long species =
        idx / nvox;

    float p =
        1.0f
        - expf(-rates[species] * dt);

    unsigned int survivors = 0;

    for (unsigned int m = 0; m < n; m++)
    {
        unsigned long long key =
            ((unsigned long long)idx
             * 0xd6e8feb86659fd93ULL)
            ^ ((unsigned long long)step
             * 0xa5a35625aae3e29dULL)
            ^ m;

        if (random01(key) >= p)
            survivors++;
    }

    counts[idx] = survivors;
}


// ============================================================
// GENERIC COUNT-BASED CATALYTIC REACTION
//
// enzyme + substrate -> enzyme + product
//
// Useful computational pattern for:
//
// kinase
// phosphatase
// acetylase
// enzyme conversion
// etc.
// ============================================================

extern "C" __global__
void count_catalytic(
    unsigned int* counts,
    const int* enzyme,
    const int* substrate,
    const int* product,
    const float* rates,
    const unsigned char* mask,
    long long nvox,
    long long total,
    float dt,
    unsigned long long step)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    long long pair =
        idx / nvox;

    long long voxel =
        idx % nvox;

    if (!mask[voxel])
        return;

    int e = enzyme[pair];
    int s = substrate[pair];
    int p = product[pair];

    unsigned int ne =
        counts[(long long)e * nvox + voxel];

    unsigned int ns =
        counts[(long long)s * nvox + voxel];

    if (ne == 0 || ns == 0)
        return;

    float probability =
        1.0f
        - expf(-rates[pair]
               * (float)ne
               * dt);

    unsigned int events = 0;

    for (unsigned int m = 0; m < ns; m++)
    {
        unsigned long long key =
            ((unsigned long long)idx
             * 0x9e3779b97f4a7c15ULL)
            ^ ((unsigned long long)step
             * 0x94d049bb133111ebULL)
            ^ m;

        if (random01(key) < probability)
            events++;
    }

    counts[
        (long long)s * nvox
        + voxel
    ] -= events;

    counts[
        (long long)p * nvox
        + voxel
    ] += events;
}


// ============================================================
// GENERIC CONCENTRATION CATALYTIC REACTION
//
// E + S -> E + P
// ============================================================

extern "C" __global__
void field_catalytic(
    float* fields,
    const int* enzyme,
    const int* substrate,
    const int* product,
    const float* rates,
    const unsigned char* mask,
    long long nvox,
    long long total,
    float dt)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    long long pair =
        idx / nvox;

    long long voxel =
        idx % nvox;

    if (!mask[voxel])
        return;

    int e = enzyme[pair];
    int s = substrate[pair];
    int p = product[pair];

    float E =
        fields[
            (long long)e * nvox
            + voxel
        ];

    float S =
        fields[
            (long long)s * nvox
            + voxel
        ];

    float flux =
        rates[pair]
        * E
        * S
        * dt;

    if (flux > S)
        flux = S;

    fields[
        (long long)s * nvox
        + voxel
    ] -= flux;

    fields[
        (long long)p * nvox
        + voxel
    ] += flux;
}


// ============================================================
// COUNT + CONCENTRATION REACTION
//
// discrete enzyme + concentration substrate
//       -> discrete enzyme + concentration product
//
// This is important for interactions between rare proteins
// and abundant metabolites.
// ============================================================

extern "C" __global__
void mixed_catalytic(
    const unsigned int* counts,
    float* fields,
    const int* enzyme,
    const int* substrate,
    const int* product,
    const float* rates,
    const unsigned char* mask,
    long long nvox,
    long long total,
    float dt)
{
    long long idx =
        (long long)blockIdx.x * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;

    long long pair =
        idx / nvox;

    long long voxel =
        idx % nvox;

    if (!mask[voxel])
        return;

    int e = enzyme[pair];
    int s = substrate[pair];
    int p = product[pair];

    unsigned int E =
        counts[
            (long long)e * nvox
            + voxel
        ];

    if (E == 0)
        return;

    float S =
        fields[
            (long long)s * nvox
            + voxel
        ];

    float flux =
        rates[pair]
        * (float)E
        * S
        * dt;

    if (flux > S)
        flux = S;

    fields[
        (long long)s * nvox
        + voxel
    ] -= flux;

    fields[
        (long long)p * nvox
        + voxel
    ] += flux;
}

'''


module = cp.RawModule(code=CUDA_SRC)

diffuse_counts = module.get_function(
    "diffuse_counts"
)

diffuse_fields = module.get_function(
    "diffuse_fields"
)

count_decay = module.get_function(
    "count_decay"
)

count_catalytic = module.get_function(
    "count_catalytic"
)

field_catalytic = module.get_function(
    "field_catalytic"
)

mixed_catalytic = module.get_function(
    "mixed_catalytic"
)


# ============================================================
# State allocation
# ============================================================

allocation_start = time.perf_counter()

COUNT_SPECIES = args.count_species
FIELD_SPECIES = args.field_species

count_values = COUNT_SPECIES * NVOX
field_values = FIELD_SPECIES * NVOX

counts_a = cp.zeros(
    count_values,
    dtype=cp.uint32
)

counts_b = cp.zeros_like(counts_a)

fields_a = cp.ones(
    field_values,
    dtype=cp.float32
)

fields_b = cp.empty_like(fields_a)

# Remove concentration outside the cell
fields_a = fields_a.reshape(
    FIELD_SPECIES,
    NVOX
)

fields_a *= mask[None, :]

fields_a = fields_a.reshape(-1)

# ------------------------------------------------------------
# Randomly distribute count molecules across intracellular
# voxels and count species.
# ------------------------------------------------------------

cell_indices = cp.asarray(
    np.flatnonzero(mask_cpu.reshape(-1)),
    dtype=cp.int64
)

particle_species = cp.random.randint(
    0,
    COUNT_SPECIES,
    args.particles,
    dtype=cp.int64
)

particle_voxels = cell_indices[
    cp.random.randint(
        0,
        CELL_VOXELS,
        args.particles
    )
]

particle_index = (
    particle_species * NVOX
    + particle_voxels
)

cp.add.at(
    counts_a,
    particle_index,
    1
)

# Species-specific degradation rates
decay_rates = cp.linspace(
    0.05,
    0.20,
    COUNT_SPECIES,
    dtype=cp.float32
)


# ============================================================
# Generic reaction lookup tables
# ============================================================

count_pairs = min(
    args.reaction_pairs,
    COUNT_SPECIES // 3
)

field_pairs = min(
    args.reaction_pairs,
    FIELD_SPECIES // 3
)

mixed_pairs = min(
    args.reaction_pairs,
    COUNT_SPECIES,
    FIELD_SPECIES // 2
)


# Count reactions:
# species 0 catalyzes 1 -> 2
# species 3 catalyzes 4 -> 5
# ...

ce = cp.arange(
    count_pairs,
    dtype=cp.int32
) * 3

cs = ce + 1
cp_ = ce + 2

crates = cp.linspace(
    1.0,
    5.0,
    count_pairs,
    dtype=cp.float32
)


# Concentration reactions

fe = cp.arange(
    field_pairs,
    dtype=cp.int32
) * 3

fs = fe + 1
fp = fe + 2

frates = cp.linspace(
    0.05,
    0.20,
    field_pairs,
    dtype=cp.float32
)


# Mixed reactions:
# count enzyme acts on concentration substrate

me = cp.arange(
    mixed_pairs,
    dtype=cp.int32
)

ms = cp.arange(
    mixed_pairs,
    dtype=cp.int32
) * 2

mp = ms + 1

mrates = cp.linspace(
    0.05,
    0.20,
    mixed_pairs,
    dtype=cp.float32
)


cp.cuda.Stream.null.synchronize()

allocation_time = (
    time.perf_counter()
    - allocation_start
)


# ============================================================
# Kernel launch configuration
# ============================================================

THREADS = 256


def blocks(total):
    return (
        int(
            (total + THREADS - 1)
            // THREADS
        ),
    )


# ============================================================
# Warmup / CUDA compilation
# ============================================================

counts_b.fill(0)

diffuse_counts(
    blocks(count_values),
    (THREADS,),
    (
        counts_a,
        counts_b,
        mask,
        np.int32(NX),
        np.int32(NY),
        np.int32(NZ),
        np.int64(NVOX),
        np.int64(count_values),
        np.float32(q),
        np.uint64(0)
    )
)

diffuse_fields(
    blocks(field_values),
    (THREADS,),
    (
        fields_a,
        fields_b,
        mask,
        np.int32(NX),
        np.int32(NY),
        np.int32(NZ),
        np.int64(NVOX),
        np.int64(field_values),
        np.float32(q)
    )
)

cp.cuda.Stream.null.synchronize()


# ============================================================
# Timing helper
# ============================================================

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
# COUNT DIFFUSION
# ============================================================

def benchmark_count_diffusion():

    global counts_a, counts_b

    for step in range(args.steps):

        counts_b.fill(0)

        diffuse_counts(
            blocks(count_values),
            (THREADS,),
            (
                counts_a,
                counts_b,
                mask,
                np.int32(NX),
                np.int32(NY),
                np.int32(NZ),
                np.int64(NVOX),
                np.int64(count_values),
                np.float32(q),
                np.uint64(step)
            )
        )

        counts_a, counts_b = (
            counts_b,
            counts_a
        )


count_diff_ms = gpu_time(
    benchmark_count_diffusion
)


# ============================================================
# FIELD DIFFUSION
# ============================================================

def benchmark_field_diffusion():

    global fields_a, fields_b

    for _ in range(args.steps):

        diffuse_fields(
            blocks(field_values),
            (THREADS,),
            (
                fields_a,
                fields_b,
                mask,
                np.int32(NX),
                np.int32(NY),
                np.int32(NZ),
                np.int64(NVOX),
                np.int64(field_values),
                np.float32(q)
            )
        )

        fields_a, fields_b = (
            fields_b,
            fields_a
        )


field_diff_ms = gpu_time(
    benchmark_field_diffusion
)


# ============================================================
# COUNT DEGRADATION
# ============================================================

def benchmark_decay():

    for step in range(args.steps):

        count_decay(
            blocks(count_values),
            (THREADS,),
            (
                counts_a,
                decay_rates,
                np.int64(NVOX),
                np.int64(count_values),
                np.float32(dt),
                np.uint64(step)
            )
        )


decay_ms = gpu_time(
    benchmark_decay
)


# ============================================================
# COUNT CATALYTIC REACTIONS
# ============================================================

count_reaction_total = (
    count_pairs * NVOX
)


def benchmark_count_reactions():

    for step in range(args.steps):

        count_catalytic(
            blocks(count_reaction_total),
            (THREADS,),
            (
                counts_a,
                ce,
                cs,
                cp_,
                crates,
                mask,
                np.int64(NVOX),
                np.int64(count_reaction_total),
                np.float32(dt),
                np.uint64(step)
            )
        )


count_rxn_ms = gpu_time(
    benchmark_count_reactions
)


# ============================================================
# FIELD CATALYTIC REACTIONS
# ============================================================

field_reaction_total = (
    field_pairs * NVOX
)


def benchmark_field_reactions():

    for _ in range(args.steps):

        field_catalytic(
            blocks(field_reaction_total),
            (THREADS,),
            (
                fields_a,
                fe,
                fs,
                fp,
                frates,
                mask,
                np.int64(NVOX),
                np.int64(field_reaction_total),
                np.float32(dt)
            )
        )


field_rxn_ms = gpu_time(
    benchmark_field_reactions
)


# ============================================================
# MIXED COUNT + FIELD REACTIONS
# ============================================================

mixed_reaction_total = (
    mixed_pairs * NVOX
)


def benchmark_mixed_reactions():

    for _ in range(args.steps):

        mixed_catalytic(
            blocks(mixed_reaction_total),
            (THREADS,),
            (
                counts_a,
                fields_a,
                me,
                ms,
                mp,
                mrates,
                mask,
                np.int64(NVOX),
                np.int64(mixed_reaction_total),
                np.float32(dt)
            )
        )


mixed_rxn_ms = gpu_time(
    benchmark_mixed_reactions
)


# ============================================================
# Results
# ============================================================

pool = cp.get_default_memory_pool()

memory_used = (
    pool.used_bytes()
    / 1024**3
)

memory_reserved = (
    pool.total_bytes()
    / 1024**3
)


count_diff_per = (
    count_diff_ms
    / args.steps
)

field_diff_per = (
    field_diff_ms
    / args.steps
)

decay_per = (
    decay_ms
    / args.steps
)

count_rxn_per = (
    count_rxn_ms
    / args.steps
)

field_rxn_per = (
    field_rxn_ms
    / args.steps
)

mixed_rxn_per = (
    mixed_rxn_ms
    / args.steps
)

combined_ms = (
    count_diff_per
    + field_diff_per
    + decay_per
    + count_rxn_per
    + field_rxn_per
    + mixed_rxn_per
)

steps_per_second_bio = (
    1.0 / dt
)

wall_per_bio_second = (
    combined_ms
    / 1000.0
    * steps_per_second_bio
)

wall_40min_hours = (
    wall_per_bio_second
    * 2400
    / 3600
)


print()
print("RESULTS")
print("--------------------------------------------")

print(
    f"Allocation time:             "
    f"{allocation_time:.3f} s"
)

print(
    f"GPU memory used:             "
    f"{memory_used:.2f} GiB"
)

print(
    f"GPU memory reserved:         "
    f"{memory_reserved:.2f} GiB"
)

print()

print(
    f"Count diffusion/update:      "
    f"{count_diff_per:.3f} ms"
)

print(
    f"Field diffusion/update:      "
    f"{field_diff_per:.3f} ms"
)

print(
    f"Count degradation/update:    "
    f"{decay_per:.3f} ms"
)

print(
    f"Count catalytic/update:      "
    f"{count_rxn_per:.3f} ms"
)

print(
    f"Field catalytic/update:      "
    f"{field_rxn_per:.3f} ms"
)

print(
    f"Mixed catalytic/update:      "
    f"{mixed_rxn_per:.3f} ms"
)

print()

print(
    f"Combined/update:             "
    f"{combined_ms:.3f} ms"
)

print(
    f"Updates / biological sec:    "
    f"{steps_per_second_bio:,.0f}"
)

print(
    f"Projected wall / bio sec:    "
    f"{wall_per_bio_second:.3f} s"
)

print(
    f"Projected 40-min cell cycle: "
    f"{wall_40min_hours:.2f} wall hours"
)

print()

print(
    f"Final molecule count:        "
    f"{int(cp.sum(counts_a)):,}"
)

print(
    f"Final field checksum:        "
    f"{float(cp.sum(fields_a)):.3e}"
)
