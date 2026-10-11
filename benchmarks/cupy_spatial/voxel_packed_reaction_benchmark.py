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
parser.add_argument("--max-slots", type=int, default=128)

parser.add_argument(
    "--field-species",
    type=int,
    default=512,
    help="Dense concentration-field species"
)

parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--D", type=float, default=1.0)

args = parser.parse_args()


# ============================================================
# Geometry
#
# Everything derives from voxel size.
# 50 nm is the DEFAULT, not hard-coded into the simulator.
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
# Compact bounding voxel -> intracellular voxel map
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
# Timestep derived from voxel size
# ============================================================

q = 0.5 / 6.0

dt = (
    q
    * spacing_um**2
    / args.D
)


# ============================================================
# Initial particle state
# ============================================================

if args.species > 65534:

    raise ValueError(
        "uint16 packed representation currently supports "
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


# ------------------------------------------------------------
# Add some extra catalyst / partner molecules.
#
# We deliberately bias a fraction of molecules toward a small
# catalyst pool so that particle-particle reaction classes
# actually fire during this short benchmark.
# ------------------------------------------------------------

catalyst_pool_size = min(
    64,
    args.species
)

partner_pool_size = min(
    64,
    args.species
)

biased = min(
    args.molecules // 10,
    25000
)

if biased > 0:

    initial_species[:biased] = rng.integers(
        1,
        catalyst_pool_size + 1,
        size=biased,
        dtype=np.uint16
    )


# ============================================================
# Pack molecules into voxel slots
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
        f"Initial occupancy {initial_max_occupancy} "
        f"exceeds max slots {args.max_slots}."
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

slot_sorted = (
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
    slot_sorted
] = species_sorted


# ============================================================
# Reaction metadata
#
# reaction class:
#
# 0 = none
# 1 = degradation
# 2 = first-order conversion
# 3 = catalyst + substrate -> catalyst + product
# 4 = binding
# 5 = unbinding
# 6 = particle + concentration-field catalytic reaction
#
# IMPORTANT:
# This is the key architecture.
#
# We store reaction BEHAVIOR with the species rather than
# making a giant global reaction list.
# ============================================================

N = args.species + 1

reaction_class_cpu = np.zeros(
    N,
    dtype=np.uint8
)


# ------------------------------------------------------------
# Class 1: degradation
# ------------------------------------------------------------

decay_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)


# ------------------------------------------------------------
# Class 2: simple conversion
#
# A -> B
# ------------------------------------------------------------

conversion_product_cpu = np.zeros(
    N,
    dtype=np.uint16
)

conversion_rate_cpu = np.zeros(
    N,
    dtype=np.float32
)


# ------------------------------------------------------------
# Class 3: catalytic particle reaction
#
# enzyme + substrate
#       ->
# enzyme + product
#
# Metadata stored on SUBSTRATE.
# ------------------------------------------------------------

catalyst_species_cpu = np.zeros(
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


# ------------------------------------------------------------
# Class 4: binding
#
# A + B -> complex
#
# Metadata stored on A.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Class 5: unbinding
#
# complex -> A + B
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Class 6: particle + concentration
#
# enzyme particle + field substrate
#       ->
# enzyme particle + field product
# ------------------------------------------------------------

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
# Assign synthetic reaction classes
#
# We use blocks based on species ID.
#
# The exact biology does NOT matter here.
# We are testing computational architecture.
# ============================================================

for s in range(
    1,
    args.species + 1
):

    category = s % 12

    # --------------------------------------------------------
    # degradation
    # --------------------------------------------------------

    if category == 1:

        reaction_class_cpu[s] = 1

        decay_rate_cpu[s] = (
            0.25
            + 0.01 * (s % 10)
        )


    # --------------------------------------------------------
    # first-order conversion
    # --------------------------------------------------------

    elif (
        category == 2
        and s + 1 <= args.species
    ):

        reaction_class_cpu[s] = 2

        conversion_product_cpu[s] = (
            s + 1
        )

        conversion_rate_cpu[s] = (
            0.5
            + 0.02 * (s % 10)
        )


    # --------------------------------------------------------
    # catalytic / kinase-like
    # --------------------------------------------------------

    elif (
        category == 4
        and s + 1 <= args.species
    ):

        reaction_class_cpu[s] = 3

        catalyst_species_cpu[s] = (
            1
            + (
                (s * 17)
                % catalyst_pool_size
            )
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
    #
    # species s + selected partner
    # -> s+1
    #
    # s+1 becomes an unbinding complex.
    # --------------------------------------------------------

    elif (
        category == 6
        and s + 1 <= args.species
    ):

        reaction_class_cpu[s] = 4

        partner = (
            1
            + (
                (s * 31)
                % partner_pool_size
            )
        )

        bind_partner_cpu[s] = partner
        bind_product_cpu[s] = s + 1
        bind_rate_cpu[s] = 1.5

        # s+1 is the complex
        reaction_class_cpu[s + 1] = 5

        unbind_a_cpu[s + 1] = s
        unbind_b_cpu[s + 1] = partner
        unbind_rate_cpu[s + 1] = 0.5


    # --------------------------------------------------------
    # particle + concentration field
    # --------------------------------------------------------

    elif (
        category == 8
        and args.field_species >= 2
    ):

        reaction_class_cpu[s] = 6

        substrate_field = (
            (s * 7)
            % (args.field_species - 1)
        )

        field_substrate_cpu[s] = (
            substrate_field
        )

        field_product_cpu[s] = (
            substrate_field + 1
        )

        field_rate_cpu[s] = (
            0.1
            + 0.01 * (s % 10)
        )


# ============================================================
# Move metadata to GPU
# ============================================================

reaction_class = cp.asarray(
    reaction_class_cpu
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

catalyst_species = cp.asarray(
    catalyst_species_cpu
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
# Concentration fields
# ============================================================

fields_initial = cp.ones(
    (
        args.field_species,
        CELL_VOXELS
    ),
    dtype=cp.float32
)


# ============================================================
# CUDA source
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
// Packed diffusion
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

    unsigned short s =
        species_in[
            (long long)voxel
            * max_slots
            + slot
        ];

    if (s == 0)
        return;


    unsigned long long key =
        ((unsigned long long)idx
         * 0x9e3779b97f4a7c15ULL)
        ^
        ((unsigned long long)step
         * 0xbf58476d1ce4e5b9ULL);

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

    unsigned int destination_slot =
        atomicAdd(
            &occupancy_out[destination],
            1U
        );

    if (
        destination_slot
        < (unsigned int)max_slots
    )
    {
        species_out[
            (long long)destination
            * max_slots
            + destination_slot
        ] = s;
    }

    else
    {
        atomicAdd(
            overflow,
            1U
        );
    }
}


// ============================================================
// Generic reaction engine
//
// ONE THREAD = ONE VOXEL
//
// Therefore reactions within one voxel are handled serially,
// avoiding races between particles in the same voxel.
//
// The particle's species ID determines which reaction class
// and species-specific parameters are used.
// ============================================================

extern "C" __global__
void reaction_engine(
    unsigned short* particles,
    unsigned int* occupancy,

    float* fields,

    const unsigned char* reaction_class,

    const float* decay_rate,

    const unsigned short* conversion_product,
    const float* conversion_rate,

    const unsigned short* catalyst_species,
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


    unsigned int local_events[6];

    #pragma unroll
    for (int e = 0; e < 6; e++)
        local_events[e] = 0;


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


        unsigned char cls =
            reaction_class[s];


        unsigned long long key =
            ((unsigned long long)voxel
             * 0x9e3779b97f4a7c15ULL)
            ^
            ((unsigned long long)i
             * 0xbf58476d1ce4e5b9ULL)
            ^
            ((unsigned long long)step
             * 0x94d049bb133111ebULL)
            ^
            ((unsigned long long)cls
             * 0xd6e8feb86659fd93ULL);


        // ====================================================
        // CLASS 1
        // degradation
        // ====================================================

        if (cls == 1)
        {
            float p =
                1.0f
                - expf(
                    -decay_rate[s]
                    * dt
                );

            if (random01(key) < p)
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

                // Do not increment i.
                // We need to inspect the particle that was
                // swapped into this slot.

                continue;
            }
        }


        // ====================================================
        // CLASS 2
        // first-order state conversion
        // ====================================================

        else if (cls == 2)
        {
            float p =
                1.0f
                - expf(
                    -conversion_rate[s]
                    * dt
                );

            if (random01(key) < p)
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
            }
        }


        // ====================================================
        // CLASS 3
        //
        // catalyst + substrate
        //      ->
        // catalyst + product
        //
        // Search ONLY this voxel for the catalyst.
        // ====================================================

        else if (cls == 3)
        {
            unsigned short catalyst =
                catalyst_species[s];

            unsigned int catalyst_count = 0;

            for (
                unsigned int j = 0;
                j < n;
                j++
            )
            {
                if (
                    particles[
                        base + j
                    ] == catalyst
                )
                {
                    catalyst_count++;
                }
            }

            if (catalyst_count > 0)
            {
                float p =
                    1.0f
                    - expf(
                        -catalytic_rate[s]
                        * catalyst_count
                        * dt
                    );

                if (random01(key) < p)
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
                }
            }
        }


        // ====================================================
        // CLASS 4
        //
        // binding:
        //
        // A + B -> complex
        //
        // Search only later slots so removing the partner
        // doesn't invalidate already-processed slots.
        // ====================================================

        else if (cls == 4)
        {
            unsigned short partner =
                bind_partner[s];

            int partner_slot = -1;

            for (
                unsigned int j = i + 1;
                j < n;
                j++
            )
            {
                if (
                    particles[
                        base + j
                    ] == partner
                )
                {
                    partner_slot = j;
                    break;
                }
            }

            if (partner_slot >= 0)
            {
                float p =
                    1.0f
                    - expf(
                        -bind_rate[s]
                        * dt
                    );

                if (random01(key) < p)
                {
                    unsigned short complex =
                        bind_product[s];

                    particles[
                        base + i
                    ] = complex;

                    n--;

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
                }
            }
        }


        // ====================================================
        // CLASS 5
        //
        // unbinding:
        //
        // complex -> A + B
        //
        // If the primary voxel slots are full we skip it.
        // Proper overflow handling is the NEXT to-do item.
        // ====================================================

        else if (cls == 5)
        {
            if (
                n
                < (unsigned int)max_slots
            )
            {
                float p =
                    1.0f
                    - expf(
                        -unbind_rate[s]
                        * dt
                    );

                if (random01(key) < p)
                {
                    unsigned short a =
                        unbind_a[s];

                    unsigned short b =
                        unbind_b[s];

                    particles[
                        base + i
                    ] = a;

                    particles[
                        base + n
                    ] = b;

                    n++;

                    local_events[4]++;
                }
            }
        }


        // ====================================================
        // CLASS 6
        //
        // discrete particle + continuous field
        //
        // enzyme_particle + substrate_field
        //      ->
        // enzyme_particle + product_field
        // ====================================================

        else if (cls == 6)
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
                long long substrate_index =
                    (long long)fs
                    * nvox
                    + voxel;

                long long product_index =
                    (long long)fp
                    * nvox
                    + voxel;

                float S =
                    fields[
                        substrate_index
                    ];

                if (S > 0.0f)
                {
                    float flux =
                        field_rate[s]
                        * S
                        * dt;

                    if (flux > S)
                        flux = S;

                    fields[
                        substrate_index
                    ] -= flux;

                    fields[
                        product_index
                    ] += flux;

                    local_events[5]++;
                }
            }
        }


        i++;
    }


    occupancy[voxel] = n;


    // Each voxel owns its own event counters,
    // so no global atomics are needed.

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

packed_diffuse = module.get_function(
    "packed_diffuse"
)

reaction_engine = module.get_function(
    "reaction_engine"
)


# ============================================================
# GPU setup
# ============================================================

THREADS = 256

TOTAL_SLOTS = (
    CELL_VOXELS
    * args.max_slots
)


def blocks(n):

    return (
        int(
            (n + THREADS - 1)
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
# State reset helper
# ============================================================

def make_particle_state():

    particles_a = cp.asarray(
        packed_cpu.reshape(-1)
    )

    occupancy_a = cp.asarray(
        occupancy_cpu
    )

    particles_b = cp.zeros_like(
        particles_a
    )

    occupancy_b = cp.zeros_like(
        occupancy_a
    )

    fields = fields_initial.copy()

    events = cp.zeros(
        CELL_VOXELS * 6,
        dtype=cp.uint32
    )

    overflow = cp.zeros(
        1,
        dtype=cp.uint32
    )

    return (
        particles_a,
        occupancy_a,
        particles_b,
        occupancy_b,
        fields,
        events,
        overflow
    )


# ============================================================
# Warmup
# ============================================================

(
    pa,
    oa,
    pb,
    ob,
    fields,
    events,
    overflow
) = make_particle_state()


reaction_engine(
    blocks(CELL_VOXELS),
    (THREADS,),
    (
        pa,
        oa,

        fields,

        reaction_class,

        decay_rate,

        conversion_product,
        conversion_rate,

        catalyst_species,
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

        np.int32(
            args.max_slots
        ),

        np.int64(
            CELL_VOXELS
        ),

        np.float32(dt),

        np.uint64(0)
    )
)

cp.cuda.Stream.null.synchronize()


# ============================================================
# Benchmark 1:
# reaction engine only
# ============================================================

(
    pa,
    oa,
    pb,
    ob,
    fields,
    events,
    overflow
) = make_particle_state()


def run_reactions():

    for step in range(
        args.steps
    ):

        reaction_engine(
            blocks(CELL_VOXELS),
            (THREADS,),
            (
                pa,
                oa,

                fields,

                reaction_class,

                decay_rate,

                conversion_product,
                conversion_rate,

                catalyst_species,
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

                np.int32(
                    args.max_slots
                ),

                np.int64(
                    CELL_VOXELS
                ),

                np.float32(dt),

                np.uint64(step)
            )
        )


reaction_ms = gpu_time(
    run_reactions
)

reaction_events = (
    events
    .reshape(
        CELL_VOXELS,
        6
    )
    .sum(
        axis=0
    )
    .get()
)

reaction_final_molecules = int(
    cp.sum(oa)
)

reaction_max_occupancy = int(
    cp.max(oa)
)


# ============================================================
# Benchmark 2:
# diffusion only
# ============================================================

(
    pa,
    oa,
    pb,
    ob,
    fields,
    events,
    overflow
) = make_particle_state()


def run_diffusion():

    global pa, oa, pb, ob

    for step in range(
        args.steps
    ):

        pb.fill(0)
        ob.fill(0)

        packed_diffuse(
            blocks(TOTAL_SLOTS),
            (THREADS,),
            (
                pa,
                oa,

                pb,
                ob,

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

        pa, pb = pb, pa
        oa, ob = ob, oa


diffusion_ms = gpu_time(
    run_diffusion
)


# ============================================================
# Benchmark 3:
# diffusion + reaction cycle
# ============================================================

(
    pa,
    oa,
    pb,
    ob,
    fields,
    events,
    overflow
) = make_particle_state()


def run_combined():

    global pa, oa, pb, ob

    for step in range(
        args.steps
    ):

        # ----------------------------------------------------
        # Diffusion
        # ----------------------------------------------------

        pb.fill(0)
        ob.fill(0)

        packed_diffuse(
            blocks(TOTAL_SLOTS),
            (THREADS,),
            (
                pa,
                oa,

                pb,
                ob,

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

        pa, pb = pb, pa
        oa, ob = ob, oa


        # ----------------------------------------------------
        # Reactions
        # ----------------------------------------------------

        reaction_engine(
            blocks(CELL_VOXELS),
            (THREADS,),
            (
                pa,
                oa,

                fields,

                reaction_class,

                decay_rate,

                conversion_product,
                conversion_rate,

                catalyst_species,
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

                np.int32(
                    args.max_slots
                ),

                np.int64(
                    CELL_VOXELS
                ),

                np.float32(dt),

                np.uint64(step)
            )
        )


combined_ms = gpu_time(
    run_combined
)

combined_events = (
    events
    .reshape(
        CELL_VOXELS,
        6
    )
    .sum(
        axis=0
    )
    .get()
)

combined_final_molecules = int(
    cp.sum(oa)
)

combined_max_occupancy = int(
    cp.max(oa)
)

overflow_count = int(
    overflow[0]
)


# ============================================================
# Memory
# ============================================================

pool = cp.get_default_memory_pool()

particle_buffer_bytes = (
    pa.nbytes
    + pb.nbytes
    + oa.nbytes
    + ob.nbytes
)

field_bytes = fields.nbytes


# ============================================================
# Output
# ============================================================

print()
print("CONFIGURATION")
print("--------------------------------------------")

print(
    f"Voxel size:               "
    f"{args.spacing_nm:.1f} nm"
)

print(
    f"Cell voxels:              "
    f"{CELL_VOXELS:,}"
)

print(
    f"Species universe:         "
    f"{args.species:,}"
)

print(
    f"Initial molecules:        "
    f"{args.molecules:,}"
)

print(
    f"Primary slots / voxel:    "
    f"{args.max_slots}"
)

print(
    f"Initial max occupancy:    "
    f"{initial_max_occupancy}"
)

print(
    f"Concentration fields:     "
    f"{args.field_species:,}"
)

print(
    f"Timestep:                 "
    f"{dt * 1e6:.2f} us"
)


print()
print("MEMORY")
print("--------------------------------------------")

print(
    f"Particle buffers:         "
    f"{particle_buffer_bytes / 1024**2:.2f} MiB"
)

print(
    f"Concentration fields:     "
    f"{field_bytes / 1024**2:.2f} MiB"
)

print(
    f"CuPy memory used:         "
    f"{pool.used_bytes() / 1024**2:.2f} MiB"
)

print(
    f"CuPy memory reserved:     "
    f"{pool.total_bytes() / 1024**2:.2f} MiB"
)


print()
print("PERFORMANCE")
print("--------------------------------------------")

print(
    f"Diffusion / update:       "
    f"{diffusion_ms / args.steps:.4f} ms"
)

print(
    f"Reaction engine / update: "
    f"{reaction_ms / args.steps:.4f} ms"
)

print(
    f"Combined / update:        "
    f"{combined_ms / args.steps:.4f} ms"
)


print()
print("REACTION-ONLY EVENTS")
print("--------------------------------------------")

labels = [
    "Degradation",
    "Conversion",
    "Catalytic",
    "Binding",
    "Unbinding",
    "Particle + field"
]

for label, value in zip(
    labels,
    reaction_events
):

    print(
        f"{label:22s}"
        f"{int(value):,}"
    )


print()
print("COMBINED EVENTS")
print("--------------------------------------------")

for label, value in zip(
    labels,
    combined_events
):

    print(
        f"{label:22s}"
        f"{int(value):,}"
    )


print()
print("FINAL STATE")
print("--------------------------------------------")

print(
    f"Reaction-only molecules:  "
    f"{reaction_final_molecules:,}"
)

print(
    f"Reaction-only max occ:    "
    f"{reaction_max_occupancy}"
)

print(
    f"Combined molecules:       "
    f"{combined_final_molecules:,}"
)

print(
    f"Combined max occupancy:   "
    f"{combined_max_occupancy}"
)

print(
    f"Diffusion overflow events:"
    f" {overflow_count:,}"
)

print(
    f"Field checksum:           "
    f"{float(cp.sum(fields)):.6e}"
)
