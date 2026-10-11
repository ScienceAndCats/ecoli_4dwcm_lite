import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp


# ============================================================
# Import our real scheduler
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from compute.multirate import MultiRateScheduler


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
    "--D-fast",
    type=float,
    default=1.0
)

parser.add_argument(
    "--D-field",
    type=float,
    default=0.1
)

parser.add_argument(
    "--reaction-interval-ms",
    type=float,
    default=1.0
)

parser.add_argument(
    "--bio-seconds",
    type=float,
    default=1.0
)

args = parser.parse_args()


# ============================================================
# Geometry
#
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

mask_flat = mask.reshape(-1)

BOUNDING_VOXELS = mask_flat.size

active_flat = np.flatnonzero(
    mask_flat
)

CELL_VOXELS = active_flat.size


# ============================================================
# Compact cell-only neighbor table
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
# Timesteps derived from voxel size + diffusion coefficient
# ============================================================

q = 0.5 / 6.0

base_dt = (
    q
    * spacing_um**2
    / args.D_fast
)

field_dt = (
    q
    * spacing_um**2
    / args.D_field
)

reaction_interval_s = (
    args.reaction_interval_ms
    / 1000.0
)


# ============================================================
# Particle initialization
# ============================================================

if args.species > 65534:
    raise ValueError(
        "uint16 species IDs currently support <=65,534 species"
    )

TOTAL_CAPACITY = (
    args.primary_slots
    + args.spill_slots
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


# Bias some molecules toward interaction partners.

PARTNER_POOL = min(
    128,
    args.species
)

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
# Sort + pack into primary/spill
# ============================================================

order = np.argsort(
    initial_voxels,
    kind="stable"
)

vox_sorted = initial_voxels[order]
species_sorted = initial_species[order]

total_occ_cpu = np.bincount(
    vox_sorted,
    minlength=CELL_VOXELS
).astype(np.uint32)

initial_max_occupancy = int(
    total_occ_cpu.max()
)

if initial_max_occupancy > TOTAL_CAPACITY:

    raise RuntimeError(
        f"Initial occupancy "
        f"{initial_max_occupancy} exceeds capacity "
        f"{TOTAL_CAPACITY}"
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
        total_occ_cpu,
        dtype=np.int64
    )
    - total_occ_cpu
)


for voxel in range(CELL_VOXELS):

    n = int(
        total_occ_cpu[voxel]
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

    np_primary = min(
        n,
        args.primary_slots
    )

    np_spill = (
        n - np_primary
    )

    primary_cpu[
        voxel,
        :np_primary
    ] = species_here[
        :np_primary
    ]

    primary_occ_cpu[voxel] = (
        np_primary
    )

    if np_spill:

        spill_cpu[
            voxel,
            :np_spill
        ] = species_here[
            np_primary:
        ]

        spill_occ_cpu[voxel] = (
            np_spill
        )


# ============================================================
# Multifunction reaction metadata
# ============================================================

CAP_DECAY = 1 << 0
CAP_CONVERT = 1 << 1
CAP_CATALYTIC = 1 << 2
CAP_BIND = 1 << 3
CAP_UNBIND = 1 << 4
CAP_FIELD = 1 << 5


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


for s in range(
    1,
    args.species + 1
):

    if s % 2 == 0:

        capabilities_cpu[s] |= CAP_DECAY

        decay_rate_cpu[s] = (
            0.2
            + 0.02 * (s % 10)
        )


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


# Transfer metadata.

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
# CUDA kernels
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


// ============================================================
// Safe destination-slot reservation
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


    atomicAdd(
        hard_overflow,
        1U
    );
}


// ============================================================
// Primary diffusion
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
// Spill diffusion
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
// Logical particle access
//
// Primary and spill are treated as one contiguous logical list
// during reactions.
// ============================================================

__device__ __forceinline__
unsigned short get_particle(
    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,
    unsigned int logical_slot,

    int primary_slots,
    int spill_slots)
{
    if (
        logical_slot
        < (unsigned int)primary_slots
    )
    {
        return primary[
            voxel
            * (long long)primary_slots
            + logical_slot
        ];
    }

    unsigned int spill_slot =
        logical_slot
        - primary_slots;

    return spill[
        voxel
        * (long long)spill_slots
        + spill_slot
    ];
}


__device__ __forceinline__
void set_particle(
    unsigned short* primary,
    unsigned short* spill,

    long long voxel,
    unsigned int logical_slot,

    unsigned short value,

    int primary_slots,
    int spill_slots)
{
    if (
        logical_slot
        < (unsigned int)primary_slots
    )
    {
        primary[
            voxel
            * (long long)primary_slots
            + logical_slot
        ] = value;
    }

    else
    {
        unsigned int spill_slot =
            logical_slot
            - primary_slots;

        spill[
            voxel
            * (long long)spill_slots
            + spill_slot
        ] = value;
    }
}


// ============================================================
// Multifunction reaction engine
//
// One GPU thread owns one voxel.
//
// This version understands both PRIMARY and SPILL particles.
// ============================================================

extern "C" __global__
void multifunction_reactions(

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

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

    int primary_slots,
    int spill_slots,

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


    unsigned int np =
        primary_occ[voxel];

    unsigned int ns =
        spill_occ[voxel];

    unsigned int n =
        np + ns;

    unsigned int capacity =
        primary_slots
        + spill_slots;


    unsigned int local_events[6] = {
        0, 0, 0, 0, 0, 0
    };


    unsigned int i = 0;


    while (i < n)
    {
        unsigned short s =
            get_particle(
                primary,
                spill,
                voxel,
                i,
                primary_slots,
                spill_slots
            );


        if (s == 0)
        {
            i++;
            continue;
        }


        unsigned char caps =
            capabilities[s];


        // ====================================================
        // Continuous particle-field flux
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

                    local_events[5]++;
                }
            }
        }


        // ====================================================
        // Discrete competing hazards
        // ====================================================

        float h_decay = 0.0f;
        float h_convert = 0.0f;
        float h_catalytic = 0.0f;
        float h_bind = 0.0f;
        float h_unbind = 0.0f;

        int partner_slot = -1;


        if (caps & CAP_DECAY)
        {
            h_decay =
                decay_rate[s];
        }


        if (caps & CAP_CONVERT)
        {
            h_convert =
                conversion_rate[s];
        }


        if (caps & CAP_CATALYTIC)
        {
            unsigned short c =
                catalyst[s];

            unsigned int catalyst_count = 0;

            for (
                unsigned int j = 0;
                j < n;
                j++
            )
            {
                if (
                    get_particle(
                        primary,
                        spill,
                        voxel,
                        j,
                        primary_slots,
                        spill_slots
                    ) == c
                )
                {
                    catalyst_count++;
                }
            }

            h_catalytic =
                catalytic_rate[s]
                * catalyst_count;
        }


        if (caps & CAP_BIND)
        {
            unsigned short partner =
                bind_partner[s];

            unsigned int partner_count = 0;


            // Only inspect later slots.
            //
            // This prevents us from deleting a particle
            // already processed during this reaction pass.

            for (
                unsigned int j = i + 1;
                j < n;
                j++
            )
            {
                if (
                    get_particle(
                        primary,
                        spill,
                        voxel,
                        j,
                        primary_slots,
                        spill_slots
                    ) == partner
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


        if (
            (caps & CAP_UNBIND)
            &&
            n < capacity
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


        if (total_hazard <= 0.0f)
        {
            i++;
            continue;
        }


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


        float chooser =
            random01(
                key
                ^ 0xd6e8feb86659fd93ULL
            )
            * total_hazard;


        // ====================================================
        // Degradation
        // ====================================================

        if (chooser < h_decay)
        {
            n--;

            unsigned short last =
                get_particle(
                    primary,
                    spill,
                    voxel,
                    n,
                    primary_slots,
                    spill_slots
                );

            set_particle(
                primary,
                spill,
                voxel,
                i,
                last,
                primary_slots,
                spill_slots
            );

            set_particle(
                primary,
                spill,
                voxel,
                n,
                0,
                primary_slots,
                spill_slots
            );


            if (ns > 0)
                ns--;

            else
                np--;


            local_events[0]++;

            continue;
        }


        chooser -= h_decay;


        // ====================================================
        // Conversion
        // ====================================================

        if (chooser < h_convert)
        {
            unsigned short product =
                conversion_product[s];

            if (product != 0)
            {
                set_particle(
                    primary,
                    spill,
                    voxel,
                    i,
                    product,
                    primary_slots,
                    spill_slots
                );

                local_events[1]++;
            }

            i++;
            continue;
        }


        chooser -= h_convert;


        // ====================================================
        // Catalytic modification
        // ====================================================

        if (chooser < h_catalytic)
        {
            unsigned short product =
                catalytic_product[s];

            if (product != 0)
            {
                set_particle(
                    primary,
                    spill,
                    voxel,
                    i,
                    product,
                    primary_slots,
                    spill_slots
                );

                local_events[2]++;
            }

            i++;
            continue;
        }


        chooser -= h_catalytic;


        // ====================================================
        // Binding
        // ====================================================

        if (
            chooser < h_bind
            &&
            partner_slot >= 0
        )
        {
            unsigned short complex =
                bind_product[s];

            set_particle(
                primary,
                spill,
                voxel,
                i,
                complex,
                primary_slots,
                spill_slots
            );


            n--;

            unsigned short last =
                get_particle(
                    primary,
                    spill,
                    voxel,
                    n,
                    primary_slots,
                    spill_slots
                );

            set_particle(
                primary,
                spill,
                voxel,
                partner_slot,
                last,
                primary_slots,
                spill_slots
            );

            set_particle(
                primary,
                spill,
                voxel,
                n,
                0,
                primary_slots,
                spill_slots
            );


            if (ns > 0)
                ns--;

            else
                np--;


            local_events[3]++;

            i++;
            continue;
        }


        // ====================================================
        // Unbinding
        // ====================================================

        if (
            h_unbind > 0.0f
            &&
            n < capacity
        )
        {
            unsigned short a =
                unbind_a[s];

            unsigned short b =
                unbind_b[s];


            set_particle(
                primary,
                spill,
                voxel,
                i,
                a,
                primary_slots,
                spill_slots
            );


            set_particle(
                primary,
                spill,
                voxel,
                n,
                b,
                primary_slots,
                spill_slots
            );


            if (
                n
                < (unsigned int)primary_slots
            )
            {
                np++;
            }

            else
            {
                ns++;
            }


            n++;

            local_events[4]++;
        }


        i++;
    }


    primary_occ[voxel] = np;
    spill_occ[voxel] = ns;


    for (int e = 0; e < 6; e++)
    {
        event_counts[
            voxel * 6
            + e
        ] += local_events[e];
    }
}


// ============================================================
// Compact-cell concentration-field diffusion
// ============================================================

extern "C" __global__
void diffuse_fields(

    const float* input,
    float* output,

    const int* neighbors,

    long long nvox,
    long long total,

    float alpha)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (idx >= total)
        return;


    long long voxel =
        idx % nvox;


    float center =
        input[idx];

    float lap =
        0.0f;


    for (int d = 0; d < 6; d++)
    {
        int neighbor =
            neighbors[
                voxel * 6
                + d
            ];

        long long neighbor_idx =
            idx
            - voxel
            + neighbor;

        lap += (
            input[neighbor_idx]
            - center
        );
    }


    output[idx] =
        center
        + alpha * lap;
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

reaction_kernel = module.get_function(
    "multifunction_reactions"
)

field_diffuse = module.get_function(
    "diffuse_fields"
)


# ============================================================
# Launch configuration
# ============================================================

THREADS = 256

PRIMARY_TOTAL = (
    CELL_VOXELS
    * args.primary_slots
)

SPILL_TOTAL = (
    CELL_VOXELS
    * args.spill_slots
)

FIELD_TOTAL = (
    CELL_VOXELS
    * args.field_species
)


def blocks(total):

    return (
        int(
            (total + THREADS - 1)
            // THREADS
        ),
    )


# ============================================================
# State factory
# ============================================================

def make_state():

    return {
        "primary_a":
            cp.asarray(
                primary_cpu.reshape(-1)
            ),

        "primary_b":
            cp.zeros(
                PRIMARY_TOTAL,
                dtype=cp.uint16
            ),

        "spill_a":
            cp.asarray(
                spill_cpu.reshape(-1)
            ),

        "spill_b":
            cp.zeros(
                SPILL_TOTAL,
                dtype=cp.uint16
            ),

        "primary_occ_a":
            cp.asarray(primary_occ_cpu),

        "primary_occ_b":
            cp.zeros(
                CELL_VOXELS,
                dtype=cp.uint32
            ),

        "spill_occ_a":
            cp.asarray(spill_occ_cpu),

        "spill_occ_b":
            cp.zeros(
                CELL_VOXELS,
                dtype=cp.uint32
            ),

        "fields_a":
            cp.ones(
                FIELD_TOTAL,
                dtype=cp.float32
            ),

        "fields_b":
            cp.empty(
                FIELD_TOTAL,
                dtype=cp.float32
            ),

        "spill_writes":
            cp.zeros(
                1,
                dtype=cp.uint32
            ),

        "hard_overflow":
            cp.zeros(
                1,
                dtype=cp.uint32
            ),

        "events":
            cp.zeros(
                CELL_VOXELS * 6,
                dtype=cp.uint32
            ),
    }


# ============================================================
# Warm up CUDA compilation
# ============================================================

state = make_state()


state[
    "primary_occ_b"
].fill(0)

state[
    "spill_occ_b"
].fill(0)


diffuse_primary(
    blocks(PRIMARY_TOTAL),
    (THREADS,),
    (
        state["primary_a"],
        state["primary_occ_a"],

        state["primary_b"],
        state["primary_occ_b"],

        state["spill_b"],
        state["spill_occ_b"],

        neighbors,

        state["spill_writes"],
        state["hard_overflow"],

        np.int32(args.primary_slots),
        np.int32(args.spill_slots),

        np.int64(CELL_VOXELS),

        np.float32(q),
        np.uint64(0)
    )
)


diffuse_spill(
    blocks(SPILL_TOTAL),
    (THREADS,),
    (
        state["spill_a"],
        state["spill_occ_a"],

        state["primary_b"],
        state["primary_occ_b"],

        state["spill_b"],
        state["spill_occ_b"],

        neighbors,

        state["spill_writes"],
        state["hard_overflow"],

        np.int32(args.primary_slots),
        np.int32(args.spill_slots),

        np.int64(CELL_VOXELS),

        np.float32(q),
        np.uint64(0)
    )
)


reaction_kernel(
    blocks(CELL_VOXELS),
    (THREADS,),
    (
        state["primary_a"],
        state["primary_occ_a"],

        state["spill_a"],
        state["spill_occ_a"],

        state["fields_a"],

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

        state["events"],

        np.int32(args.primary_slots),
        np.int32(args.spill_slots),

        np.int64(CELL_VOXELS),

        np.float32(reaction_interval_s),
        np.uint64(0)
    )
)


warm_alpha = (
    args.D_field
    * field_dt
    / spacing_um**2
)


field_diffuse(
    blocks(FIELD_TOTAL),
    (THREADS,),
    (
        state["fields_a"],
        state["fields_b"],

        neighbors,

        np.int64(CELL_VOXELS),
        np.int64(FIELD_TOTAL),

        np.float32(warm_alpha)
    )
)

cp.cuda.Stream.null.synchronize()


# ============================================================
# Fresh state for actual benchmark
# ============================================================

state = make_state()

initial_field_checksum = float(
    cp.sum(
        state["fields_a"]
    )
)


# ============================================================
# Real callbacks
# ============================================================

particle_step_counter = 0
reaction_step_counter = 0


def particle_diffusion_callback(
    elapsed_dt,
    tick,
):

    global particle_step_counter

    state[
        "primary_occ_b"
    ].fill(0)

    state[
        "spill_occ_b"
    ].fill(0)


    diffuse_primary(
        blocks(PRIMARY_TOTAL),
        (THREADS,),
        (
            state["primary_a"],
            state["primary_occ_a"],

            state["primary_b"],
            state["primary_occ_b"],

            state["spill_b"],
            state["spill_occ_b"],

            neighbors,

            state["spill_writes"],
            state["hard_overflow"],

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

            np.uint64(
                particle_step_counter
            )
        )
    )


    diffuse_spill(
        blocks(SPILL_TOTAL),
        (THREADS,),
        (
            state["spill_a"],
            state["spill_occ_a"],

            state["primary_b"],
            state["primary_occ_b"],

            state["spill_b"],
            state["spill_occ_b"],

            neighbors,

            state["spill_writes"],
            state["hard_overflow"],

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

            np.uint64(
                particle_step_counter
            )
        )
    )


    state["primary_a"], state["primary_b"] = (
        state["primary_b"],
        state["primary_a"]
    )

    state[
        "primary_occ_a"
    ], state[
        "primary_occ_b"
    ] = (
        state["primary_occ_b"],
        state["primary_occ_a"]
    )

    state["spill_a"], state["spill_b"] = (
        state["spill_b"],
        state["spill_a"]
    )

    state[
        "spill_occ_a"
    ], state[
        "spill_occ_b"
    ] = (
        state["spill_occ_b"],
        state["spill_occ_a"]
    )


    particle_step_counter += 1


def reaction_callback(
    elapsed_dt,
    tick,
):

    global reaction_step_counter

    reaction_kernel(
        blocks(CELL_VOXELS),
        (THREADS,),
        (
            state["primary_a"],
            state["primary_occ_a"],

            state["spill_a"],
            state["spill_occ_a"],

            state["fields_a"],

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

            state["events"],

            np.int32(
                args.primary_slots
            ),

            np.int32(
                args.spill_slots
            ),

            np.int64(
                CELL_VOXELS
            ),

            np.float32(
                elapsed_dt
            ),

            np.uint64(
                reaction_step_counter
            )
        )
    )


    reaction_step_counter += 1


def field_diffusion_callback(
    elapsed_dt,
    tick,
):

    alpha = (
        args.D_field
        * elapsed_dt
        / spacing_um**2
    )


    field_diffuse(
        blocks(FIELD_TOTAL),
        (THREADS,),
        (
            state["fields_a"],
            state["fields_b"],

            neighbors,

            np.int64(
                CELL_VOXELS
            ),

            np.int64(
                FIELD_TOTAL
            ),

            np.float32(alpha)
        )
    )


    state["fields_a"], state["fields_b"] = (
        state["fields_b"],
        state["fields_a"]
    )


# ============================================================
# Scheduler
# ============================================================

scheduler = MultiRateScheduler(
    base_dt_s=base_dt
)


particle_task = scheduler.add_task(
    name="particle_diffusion",
    interval_s=base_dt,
    callback=particle_diffusion_callback,
)


reaction_task = scheduler.add_task(
    name="reactions",
    interval_s=reaction_interval_s,
    callback=reaction_callback,
)


field_task = scheduler.add_task(
    name="field_diffusion",
    interval_s=field_dt,
    callback=field_diffusion_callback,
)


# ============================================================
# Integrated run
# ============================================================

gpu_start = cp.cuda.Event()
gpu_end = cp.cuda.Event()

wall_start = time.perf_counter()

gpu_start.record()

total_ticks = scheduler.run(
    args.bio_seconds
)

gpu_end.record()
gpu_end.synchronize()

wall_elapsed = (
    time.perf_counter()
    - wall_start
)

gpu_elapsed_ms = (
    cp.cuda.get_elapsed_time(
        gpu_start,
        gpu_end
    )
)


# ============================================================
# Final state
# ============================================================

event_totals = (
    state["events"]
    .reshape(
        CELL_VOXELS,
        6
    )
    .sum(axis=0)
    .get()
)


degradation_events = int(
    event_totals[0]
)

conversion_events = int(
    event_totals[1]
)

catalytic_events = int(
    event_totals[2]
)

binding_events = int(
    event_totals[3]
)

unbinding_events = int(
    event_totals[4]
)

field_flux_evaluations = int(
    event_totals[5]
)


final_primary = int(
    cp.sum(
        state["primary_occ_a"]
    )
)

final_spill = int(
    cp.sum(
        state["spill_occ_a"]
    )
)

final_particles = (
    final_primary
    + final_spill
)


expected_particles = (
    args.molecules
    - degradation_events
    - binding_events
    + unbinding_events
)


particle_difference = (
    final_particles
    - expected_particles
)


spill_writes = int(
    state["spill_writes"][0]
)

hard_overflow = int(
    state["hard_overflow"][0]
)


max_primary = int(
    cp.max(
        state["primary_occ_a"]
    )
)

max_spill = int(
    cp.max(
        state["spill_occ_a"]
    )
)


final_field_checksum = float(
    cp.sum(
        state["fields_a"]
    )
)

field_difference = (
    final_field_checksum
    - initial_field_checksum
)

field_relative_error = (
    field_difference
    / initial_field_checksum
)


# ============================================================
# Runtime projections
# ============================================================

gpu_s_per_bio_s = (
    gpu_elapsed_ms
    / 1000.0
    / args.bio_seconds
)

wall_s_per_bio_s = (
    wall_elapsed
    / args.bio_seconds
)


gpu_40min_hours = (
    gpu_s_per_bio_s
    * 2400
    / 3600
)

wall_40min_hours = (
    wall_s_per_bio_s
    * 2400
    / 3600
)


# ============================================================
# Memory
# ============================================================

pool = cp.get_default_memory_pool()


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
    f"Bounding voxels:            "
    f"{BOUNDING_VOXELS:,}"
)

print(
    f"Active cell voxels:         "
    f"{CELL_VOXELS:,}"
)

print(
    f"Particle species universe:  "
    f"{args.species:,}"
)

print(
    f"Initial particles:          "
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
    f"Initial max occupancy:      "
    f"{initial_max_occupancy}"
)

print(
    f"Concentration fields:       "
    f"{args.field_species:,}"
)

print(
    f"Biological time simulated:  "
    f"{args.bio_seconds:.3f} s"
)


print()
print("DERIVED TIMING")
print("--------------------------------------------")

print(
    f"Base timestep:              "
    f"{base_dt*1e6:.2f} us"
)

print(
    f"Base ticks:                 "
    f"{total_ticks:,}"
)

for task in scheduler.tasks:

    print(
        f"{task.name:20s} "
        f"{task.interval_ticks:6d} ticks   "
        f"{task.actual_interval_s*1000:9.4f} ms   "
        f"{task.calls:7,d} calls"
    )


print()
print("INTEGRATED PERFORMANCE")
print("--------------------------------------------")

print(
    f"GPU elapsed:                "
    f"{gpu_elapsed_ms/1000:.3f} s"
)

print(
    f"Actual wall elapsed:        "
    f"{wall_elapsed:.3f} s"
)

print(
    f"GPU sec / bio sec:          "
    f"{gpu_s_per_bio_s:.3f}"
)

print(
    f"Wall sec / bio sec:         "
    f"{wall_s_per_bio_s:.3f}"
)

print(
    f"Projected 40-min GPU time:  "
    f"{gpu_40min_hours:.2f} h"
)

print(
    f"Projected 40-min wall time: "
    f"{wall_40min_hours:.2f} h"
)


print()
print("REACTION EVENTS")
print("--------------------------------------------")

print(
    f"Degradation:                "
    f"{degradation_events:,}"
)

print(
    f"Conversion:                 "
    f"{conversion_events:,}"
)

print(
    f"Catalytic:                  "
    f"{catalytic_events:,}"
)

print(
    f"Binding:                    "
    f"{binding_events:,}"
)

print(
    f"Unbinding:                  "
    f"{unbinding_events:,}"
)

print(
    f"Particle-field flux evals:  "
    f"{field_flux_evaluations:,}"
)


print()
print("PARTICLE STATE")
print("--------------------------------------------")

print(
    f"Primary particles:          "
    f"{final_primary:,}"
)

print(
    f"Spill particles:            "
    f"{final_spill:,}"
)

print(
    f"Final particles:            "
    f"{final_particles:,}"
)

print(
    f"Expected from reactions:    "
    f"{expected_particles:,}"
)

print(
    f"Accounting difference:      "
    f"{particle_difference:+,}"
)

print(
    f"Final max primary occ:      "
    f"{max_primary}"
)

print(
    f"Final max spill occ:        "
    f"{max_spill}"
)

print(
    f"Cumulative spill writes:    "
    f"{spill_writes:,}"
)

print(
    f"HARD overflow events:       "
    f"{hard_overflow:,}"
)


print()
print("FIELD CONSERVATION")
print("--------------------------------------------")

print(
    f"Initial checksum:           "
    f"{initial_field_checksum:.6e}"
)

print(
    f"Final checksum:             "
    f"{final_field_checksum:.6e}"
)

print(
    f"Relative difference:        "
    f"{field_relative_error:.3e}"
)


print()
print("GPU MEMORY")
print("--------------------------------------------")

print(
    f"CuPy memory used:           "
    f"{pool.used_bytes()/1024**2:.2f} MiB"
)

print(
    f"CuPy memory reserved:       "
    f"{pool.total_bytes()/1024**2:.2f} MiB"
)


print()
print("VALIDATION")
print("--------------------------------------------")

particle_ok = (
    particle_difference == 0
    and hard_overflow == 0
)

field_ok = (
    abs(field_relative_error)
    < 1e-5
)

if particle_ok:
    print("PASS: particle accounting")
else:
    print("FAIL: particle accounting")

if field_ok:
    print("PASS: field conservation")
else:
    print("FAIL: field conservation")
