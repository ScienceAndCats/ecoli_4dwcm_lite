import argparse
import numpy as np
import cupy as cp


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--spacing-nm", type=float, default=50.0)
parser.add_argument("--species", type=int, default=12000)
parser.add_argument("--molecules", type=int, default=250000)
parser.add_argument("--max-slots", type=int, default=128)
parser.add_argument("--field-species", type=int, default=512)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--D", type=float, default=1.0)

args = parser.parse_args()


# ============================================================
# Geometry
#
# Everything derived from voxel size.
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

CELL_VOXELS = int(
    np.sum(mask)
)


# ============================================================
# Timestep derived from voxel size
# ============================================================

q = 0.5 / 6.0

dt = (
    q
    * spacing_um**2
    / args.D
)


# ============================================================
# Reaction capability bits
# ============================================================

CAP_DECAY = 1 << 0
CAP_CONVERT = 1 << 1
CAP_CATALYTIC = 1 << 2
CAP_BIND = 1 << 3
CAP_UNBIND = 1 << 4
CAP_FIELD = 1 << 5


# ============================================================
# Species metadata
# ============================================================

N = args.species + 1

capabilities_cpu = np.zeros(
    N,
    dtype=np.uint8
)

decay_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)

conversion_product_cpu = np.zeros(
    N,
    dtype=np.uint16
)

conversion_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)

catalyst_cpu = np.zeros(
    N,
    dtype=np.uint16
)

catalytic_product_cpu = np.zeros(
    N,
    dtype=np.uint16
)

catalytic_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)

bind_partner_cpu = np.zeros(
    N,
    dtype=np.uint16
)

bind_product_cpu = np.zeros(
    N,
    dtype=np.uint16
)

bind_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)

unbind_a_cpu = np.zeros(
    N,
    dtype=np.uint16
)

unbind_b_cpu = np.zeros(
    N,
    dtype=np.uint16
)

unbind_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)

field_substrate_cpu = np.full(
    N,
    -1,
    dtype=np.int32
)

field_product_cpu = np.full(
    N,
    -1,
    dtype=np.int32
)

field_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)


# ============================================================
# Assign MULTIPLE capabilities
#
# These intentionally overlap.
#
# s divisible by:
#
# 2  -> degradation
# 3  -> conversion
# 5  -> catalytic
# 7  -> binding
# 11 -> unbinding
# 13 -> particle-field catalysis
#
# Example:
#
# species divisible by 2, 5 and 7 has THREE capabilities.
# ============================================================

PARTNER_POOL = min(
    128,
    args.species
)


for s in range(
    1,
    args.species + 1
):

    # --------------------------------------------------------
    # degradation
    # --------------------------------------------------------

    if s % 2 == 0:

        capabilities_cpu[s] |= CAP_DECAY

        decay_rate_cpu[s] = (
            0.2
            + 0.02 * (s % 10)
        )


    # --------------------------------------------------------
    # first-order conversion
    # --------------------------------------------------------

    if (
        s % 3 == 0
        and s + 1 <= args.species
    ):

        capabilities_cpu[s] |= CAP_CONVERT

        conversion_product_cpu[s] = (
            s + 1
        )

        conversion_rate_cpu[s] = (
            0.4
            + 0.03 * (s % 10)
        )


    # --------------------------------------------------------
    # catalytic modification
    #
    # catalyst + substrate
    # ->
    # catalyst + modified substrate
    # --------------------------------------------------------

    if (
        s % 5 == 0
        and s + 1 <= args.species
    ):

        capabilities_cpu[s] |= CAP_CATALYTIC

        catalyst_cpu[s] = (
            1
            + ((s * 17) % PARTNER_POOL)
        )

        catalytic_product_cpu[s] = (
            s + 1
        )

        catalytic_rate_cpu[s] = (
            2.0
            + 0.1 * (s % 10)
        )


    # --------------------------------------------------------
    # binding
    # --------------------------------------------------------

    if (
        s % 7 == 0
        and s + 1 <= args.species
    ):

        capabilities_cpu[s] |= CAP_BIND

        bind_partner_cpu[s] = (
            1
            + ((s * 31) % PARTNER_POOL)
        )

        bind_product_cpu[s] = (
            s + 1
        )

        bind_rate_cpu[s] = 2.0


    # --------------------------------------------------------
    # unbinding
    # --------------------------------------------------------

    if (
        s % 11 == 0
        and s + 2 <= args.species
    ):

        capabilities_cpu[s] |= CAP_UNBIND

        unbind_a_cpu[s] = (
            s + 1
        )

        unbind_b_cpu[s] = (
            1
            + ((s * 19) % PARTNER_POOL)
        )

        unbind_rate_cpu[s] = 0.6


    # --------------------------------------------------------
    # particle + field reaction
    # --------------------------------------------------------

    if (
        s % 13 == 0
        and args.field_species >= 2
    ):

        capabilities_cpu[s] |= CAP_FIELD

        fs = (
            (s * 7)
            % (args.field_species - 1)
        )

        field_substrate_cpu[s] = fs
        field_product_cpu[s] = fs + 1

        field_rate_cpu[s] = (
            0.1
            + 0.01 * (s % 10)
        )


# ============================================================
# Statistics about multifunction species
# ============================================================

bit_counts = np.zeros(
    N,
    dtype=np.uint8
)

for bit in [
    CAP_DECAY,
    CAP_CONVERT,
    CAP_CATALYTIC,
    CAP_BIND,
    CAP_UNBIND,
    CAP_FIELD,
]:
    bit_counts += (
        (capabilities_cpu & bit) != 0
    )


MULTIFUNCTION_SPECIES = int(
    np.sum(bit_counts >= 2)
)

MAX_CAPABILITIES = int(
    np.max(bit_counts)
)


# ============================================================
# Initial particles
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


# Bias some particles toward the low-ID interaction pool
# so partner-dependent reactions fire in a short benchmark.

biased = min(
    args.molecules // 10,
    50000
)

initial_species[:biased] = rng.integers(
    1,
    PARTNER_POOL + 1,
    size=biased,
    dtype=np.uint16
)


# ============================================================
# Pack by voxel
# ============================================================

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
        f"Initial max occupancy "
        f"{initial_max_occupancy} > "
        f"{args.max_slots}"
    )


starts = (
    np.cumsum(
        occupancy_cpu,
        dtype=np.int64
    )
    - occupancy_cpu
)

repeated_starts = np.repeat(
    starts,
    occupancy_cpu
)

slots = (
    np.arange(
        args.molecules,
        dtype=np.int64
    )
    - repeated_starts
)


packed_cpu = np.zeros(
    (
        CELL_VOXELS,
        args.max_slots
    ),
    dtype=np.uint16
)

packed_cpu[
    vox_sorted,
    slots
] = species_sorted


# ============================================================
# Transfer metadata to GPU
# ============================================================

capabilities = cp.asarray(
    capabilities_cpu
)

decay_rate = cp.asarray(
    decay_rate_cpu
)

conversion_product = cp.asarray(
    conversion_product_cpu
)

conversion_rate = cp.asarray(
    conversion_rate_cpu
)

catalyst = cp.asarray(
    catalyst_cpu
)

catalytic_product = cp.asarray(
    catalytic_product_cpu
)

catalytic_rate = cp.asarray(
    catalytic_rate_cpu
)

bind_partner = cp.asarray(
    bind_partner_cpu
)

bind_product = cp.asarray(
    bind_product_cpu
)

bind_rate = cp.asarray(
    bind_rate_cpu
)

unbind_a = cp.asarray(
    unbind_a_cpu
)

unbind_b = cp.asarray(
    unbind_b_cpu
)

unbind_rate = cp.asarray(
    unbind_rate_cpu
)

field_substrate = cp.asarray(
    field_substrate_cpu
)

field_product = cp.asarray(
    field_product_cpu
)

field_rate = cp.asarray(
    field_rate_cpu
)


# ============================================================
# Fields
# ============================================================

fields = cp.ones(
    (
        args.field_species,
        CELL_VOXELS
    ),
    dtype=cp.float32
)


# ============================================================
# CUDA reaction kernel
# ============================================================

CUDA_SRC = r'''

#define CAP_DECAY      1
#define CAP_CONVERT    2
#define CAP_CATALYTIC  4
#define CAP_BIND       8
#define CAP_UNBIND    16
#define CAP_FIELD     32


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


extern "C" __global__
void multifunction_reactions(

    unsigned short* particles,
    unsigned int* occupancy,

    float* fields,

    const unsigned char* capabilities,

    const float* decay_rate,

    const unsigned short* conversion_product,
    const float* conversion_rate,

    const unsigned short* catalyst,
    const unsigned short* catalytic_product,
    const float* catalytic_rate,

    const unsigned short* bind_partner,
    const unsigned short* bind_product,
    const float* bind_rate,

    const unsigned short* unbind_a,
    const unsigned short* unbind_b,
    const float* unbind_rate,

    const int* field_substrate,
    const int* field_product,
    const float* field_rate,

    unsigned int* event_counts,

    int max_slots,
    long long nvox,

    float dt,
    unsigned long long step)
{
    long long voxel =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (voxel >= nvox)
        return;


    long long base =
        voxel
        * (long long)max_slots;

    unsigned int n =
        occupancy[voxel];

    if (n > (unsigned int)max_slots)
        n = max_slots;


    unsigned int local_events[6] = {
        0, 0, 0, 0, 0, 0
    };


    unsigned int i = 0;


    while (i < n)
    {
        unsigned short s =
            particles[
                base + i
            ];

        if (s == 0)
        {
            i++;
            continue;
        }


        unsigned char caps =
            capabilities[s];


        // ====================================================
        // Continuous particle + field capability
        //
        // This can occur IN ADDITION to the discrete
        // state-changing event below.
        // ====================================================

        if (caps & CAP_FIELD)
        {
            int fs =
                field_substrate[s];

            int fp =
                field_product[s];

            if (
                fs >= 0
                && fp >= 0
            )
            {
                long long si =
                    (long long)fs
                    * nvox
                    + voxel;

                long long pi =
                    (long long)fp
                    * nvox
                    + voxel;

                float S =
                    fields[si];

                if (S > 0.0f)
                {
                    float flux =
                        field_rate[s]
                        * S
                        * dt;

                    if (flux > S)
                        flux = S;

                    fields[si] -= flux;
                    fields[pi] += flux;

                    // This counts a FLUX EVALUATION,
                    // not one discrete molecule reacting.
                    local_events[5]++;
                }
            }
        }


        // ====================================================
        // Determine hazards of ALL discrete capabilities.
        // ====================================================

        float h_decay = 0.0f;
        float h_convert = 0.0f;
        float h_catalytic = 0.0f;
        float h_bind = 0.0f;
        float h_unbind = 0.0f;

        unsigned int catalyst_count = 0;
        int partner_slot = -1;


        // ----------------------------------------------------
        // degradation
        // ----------------------------------------------------

        if (caps & CAP_DECAY)
        {
            h_decay =
                decay_rate[s];
        }


        // ----------------------------------------------------
        // conversion
        // ----------------------------------------------------

        if (caps & CAP_CONVERT)
        {
            h_convert =
                conversion_rate[s];
        }


        // ----------------------------------------------------
        // catalytic modification
        // ----------------------------------------------------

        if (caps & CAP_CATALYTIC)
        {
            unsigned short c =
                catalyst[s];

            for (
                unsigned int j = 0;
                j < n;
                j++
            )
            {
                if (
                    particles[
                        base + j
                    ] == c
                )
                {
                    catalyst_count++;
                }
            }

            h_catalytic =
                catalytic_rate[s]
                * catalyst_count;
        }


        // ----------------------------------------------------
        // binding
        // ----------------------------------------------------

        if (caps & CAP_BIND)
        {
            unsigned short partner =
                bind_partner[s];

            unsigned int partner_count = 0;

            for (
                unsigned int j = 0;
                j < n;
                j++
            )
            {
                if (
                    j != i
                    &&
                    particles[
                        base + j
                    ] == partner
                )
                {
                    partner_count++;

                    if (partner_slot < 0)
                        partner_slot = j;
                }
            }

            h_bind =
                bind_rate[s]
                * partner_count;
        }


        // ----------------------------------------------------
        // unbinding
        // ----------------------------------------------------

        if (
            (caps & CAP_UNBIND)
            &&
            n < (unsigned int)max_slots
        )
        {
            h_unbind =
                unbind_rate[s];
        }


        float total_hazard =
            h_decay
            + h_convert
            + h_catalytic
            + h_bind
            + h_unbind;


        // Nothing discrete can happen.
        if (total_hazard <= 0.0f)
        {
            i++;
            continue;
        }


        // ====================================================
        // Probability that ANY discrete event happens.
        // ====================================================

        float p_any =
            1.0f
            - expf(
                -total_hazard
                * dt
            );


        unsigned long long key =
            ((unsigned long long)voxel
             * 0x9e3779b97f4a7c15ULL)
            ^
            ((unsigned long long)i
             * 0xbf58476d1ce4e5b9ULL)
            ^
            ((unsigned long long)step
             * 0x94d049bb133111ebULL);


        if (
            random01(key)
            >= p_any
        )
        {
            i++;
            continue;
        }


        // ====================================================
        // One event occurred.
        //
        // Choose WHICH competing reaction won,
        // proportional to its hazard.
        // ====================================================

        float chooser =
            random01(
                key
                ^ 0xd6e8feb86659fd93ULL
            )
            * total_hazard;


        // ----------------------------------------------------
        // degradation wins
        // ----------------------------------------------------

        if (
            chooser
            < h_decay
        )
        {
            n--;

            particles[
                base + i
            ] =
                particles[
                    base + n
                ];

            particles[
                base + n
            ] = 0;

            local_events[0]++;

            // Swapped-in particle still needs processing.
            continue;
        }


        chooser -= h_decay;


        // ----------------------------------------------------
        // conversion wins
        // ----------------------------------------------------

        if (
            chooser
            < h_convert
        )
        {
            unsigned short product =
                conversion_product[s];

            if (product != 0)
            {
                particles[
                    base + i
                ] = product;

                local_events[1]++;
            }

            i++;
            continue;
        }


        chooser -= h_convert;


        // ----------------------------------------------------
        // catalytic modification wins
        // ----------------------------------------------------

        if (
            chooser
            < h_catalytic
        )
        {
            unsigned short product =
                catalytic_product[s];

            if (product != 0)
            {
                particles[
                    base + i
                ] = product;

                local_events[2]++;
            }

            i++;
            continue;
        }


        chooser -= h_catalytic;


        // ----------------------------------------------------
        // binding wins
        // ----------------------------------------------------

        if (
            chooser
            < h_bind
            &&
            partner_slot >= 0
        )
        {
            unsigned short complex =
                bind_product[s];

            particles[
                base + i
            ] = complex;

            n--;

            // Remove partner.
            particles[
                base + partner_slot
            ] =
                particles[
                    base + n
                ];

            particles[
                base + n
            ] = 0;

            local_events[3]++;

            i++;
            continue;
        }


        // ----------------------------------------------------
        // otherwise unbinding wins
        // ----------------------------------------------------

        if (
            h_unbind > 0.0f
            &&
            n < (unsigned int)max_slots
        )
        {
            particles[
                base + i
            ] =
                unbind_a[s];

            particles[
                base + n
            ] =
                unbind_b[s];

            n++;

            local_events[4]++;
        }


        i++;
    }


    occupancy[voxel] = n;


    // Per-voxel counters avoid global atomic contention.

    for (int e = 0; e < 6; e++)
    {
        event_counts[
            voxel * 6
            + e
        ] += local_events[e];
    }
}

'''


module = cp.RawModule(
    code=CUDA_SRC
)

reaction_kernel = module.get_function(
    "multifunction_reactions"
)


# ============================================================
# GPU state
# ============================================================

particles = cp.asarray(
    packed_cpu.reshape(-1)
)

occupancy = cp.asarray(
    occupancy_cpu
)

events = cp.zeros(
    CELL_VOXELS * 6,
    dtype=cp.uint32
)


THREADS = 256


def blocks(n):
    return (
        int(
            (n + THREADS - 1)
            // THREADS
        ),
    )


# ============================================================
# Warmup
# ============================================================

reaction_kernel(
    blocks(CELL_VOXELS),
    (THREADS,),
    (
        particles,
        occupancy,
        fields,

        capabilities,

        decay_rate,

        conversion_product,
        conversion_rate,

        catalyst,
        catalytic_product,
        catalytic_rate,

        bind_partner,
        bind_product,
        bind_rate,

        unbind_a,
        unbind_b,
        unbind_rate,

        field_substrate,
        field_product,
        field_rate,

        events,

        np.int32(args.max_slots),
        np.int64(CELL_VOXELS),

        np.float32(dt),
        np.uint64(0)
    )
)

cp.cuda.Stream.null.synchronize()


# Reset state after warmup.

particles = cp.asarray(
    packed_cpu.reshape(-1)
)

occupancy = cp.asarray(
    occupancy_cpu
)

fields = cp.ones(
    (
        args.field_species,
        CELL_VOXELS
    ),
    dtype=cp.float32
)

events.fill(0)


# ============================================================
# Benchmark
# ============================================================

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

for step in range(args.steps):

    reaction_kernel(
        blocks(CELL_VOXELS),
        (THREADS,),
        (
            particles,
            occupancy,
            fields,

            capabilities,

            decay_rate,

            conversion_product,
            conversion_rate,

            catalyst,
            catalytic_product,
            catalytic_rate,

            bind_partner,
            bind_product,
            bind_rate,

            unbind_a,
            unbind_b,
            unbind_rate,

            field_substrate,
            field_product,
            field_rate,

            events,

            np.int32(args.max_slots),
            np.int64(CELL_VOXELS),

            np.float32(dt),
            np.uint64(step)
        )
    )


end.record()
end.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(
    start,
    end
)


# ============================================================
# Results
# ============================================================

event_totals = (
    events
    .reshape(
        CELL_VOXELS,
        6
    )
    .sum(axis=0)
    .get()
)

final_molecules = int(
    cp.sum(occupancy)
)

final_max_occupancy = int(
    cp.max(occupancy)
)

field_checksum = float(
    cp.sum(fields)
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
    f"Slots / voxel:             "
    f"{args.max_slots}"
)

print(
    f"Initial max occupancy:     "
    f"{initial_max_occupancy}"
)

print(
    f"Field species:             "
    f"{args.field_species:,}"
)

print(
    f"Timestep:                  "
    f"{dt*1e6:.2f} us"
)


print()
print("MULTI-FUNCTION METADATA")
print("--------------------------------------------")

print(
    f"Species with >=2 functions:"
    f" {MULTIFUNCTION_SPECIES:,}"
)

print(
    f"Max functions on one species:"
    f" {MAX_CAPABILITIES}"
)

for nfun in range(
    1,
    MAX_CAPABILITIES + 1
):

    count = int(
        np.sum(
            bit_counts == nfun
        )
    )

    print(
        f"Species with {nfun} function(s): "
        f"{count:,}"
    )


print()
print("PERFORMANCE")
print("--------------------------------------------")

print(
    f"Reaction engine / update:  "
    f"{elapsed_ms/args.steps:.4f} ms"
)


print()
print("DISCRETE REACTION EVENTS")
print("--------------------------------------------")

labels = [
    "Degradation",
    "Conversion",
    "Catalytic",
    "Binding",
    "Unbinding",
]

for label, value in zip(
    labels,
    event_totals[:5]
):

    print(
        f"{label:22s}"
        f"{int(value):,}"
    )


print()
print("CONTINUOUS FIELD WORK")
print("--------------------------------------------")

print(
    f"Particle-field flux evaluations: "
    f"{int(event_totals[5]):,}"
)


print()
print("FINAL STATE")
print("--------------------------------------------")

print(
    f"Final molecules:           "
    f"{final_molecules:,}"
)

print(
    f"Final max occupancy:       "
    f"{final_max_occupancy}"
)

print(
    f"Field checksum:            "
    f"{field_checksum:.6e}"
)

print(
    f"CuPy memory used:          "
    f"{pool.used_bytes()/1024**2:.2f} MiB"
)
