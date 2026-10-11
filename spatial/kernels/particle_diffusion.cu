__device__ __forceinline__
unsigned long long mix64(
    unsigned long long x
)
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
    unsigned long long key
)
{
    unsigned long long x =
        mix64(key);

    return (
        ((x >> 40) & 0xFFFFFFULL)
        * (1.0f / 16777216.0f)
    );
}


/*
Reserve exactly one slot in an occupancy array.

atomicCAS prevents two GPU threads from claiming
the same slot at the same time.
*/
__device__ __forceinline__
bool reserve_slot(
    unsigned int* occupancy,
    int voxel,
    unsigned int capacity,
    unsigned int* slot_out
)
{
    while (true)
    {
        unsigned int old =
            atomicAdd(
                &occupancy[voxel],
                0U
            );

        if (old >= capacity)
        {
            return false;
        }

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


/*
Try to place a particle into a destination voxel.

Order:

1. destination primary
2. destination spill
3. original/source primary
4. original/source spill

That third/fourth fallback means that if the destination
happens to fill up during a diffusion step, the particle
can simply reject the move and remain where it was.

hard_overflow should therefore be extremely unusual.
*/
__device__ __forceinline__
void place_particle(
    unsigned short species,

    int destination_voxel,
    int source_voxel,

    unsigned short* primary_out,
    unsigned int* primary_occ_out,

    unsigned short* spill_out,
    unsigned int* spill_occ_out,

    unsigned long long* spill_writes,
    unsigned long long* rejected_moves,
    unsigned long long* hard_overflow,

    int primary_slots,
    int spill_slots
)
{
    unsigned int slot;


    // ========================================================
    // Try destination primary storage
    // ========================================================

    if (
        reserve_slot(
            primary_occ_out,
            destination_voxel,
            primary_slots,
            &slot
        )
    )
    {
        primary_out[
            (long long)destination_voxel
            * primary_slots
            + slot
        ] = species;

        return;
    }


    // ========================================================
    // Try destination spill storage
    // ========================================================

    if (
        reserve_slot(
            spill_occ_out,
            destination_voxel,
            spill_slots,
            &slot
        )
    )
    {
        spill_out[
            (long long)destination_voxel
            * spill_slots
            + slot
        ] = species;

        atomicAdd(
            spill_writes,
            1ULL
        );

        return;
    }


    // ========================================================
    // Destination full.
    //
    // Reject the move and try to put the particle back into
    // its source voxel.
    // ========================================================

    if (destination_voxel != source_voxel)
    {
        atomicAdd(
            rejected_moves,
            1ULL
        );


        if (
            reserve_slot(
                primary_occ_out,
                source_voxel,
                primary_slots,
                &slot
            )
        )
        {
            primary_out[
                (long long)source_voxel
                * primary_slots
                + slot
            ] = species;

            return;
        }


        if (
            reserve_slot(
                spill_occ_out,
                source_voxel,
                spill_slots,
                &slot
            )
        )
        {
            spill_out[
                (long long)source_voxel
                * spill_slots
                + slot
            ] = species;

            atomicAdd(
                spill_writes,
                1ULL
            );

            return;
        }
    }


    // ========================================================
    // Both destination and source have no capacity.
    //
    // This is treated as a simulation error.
    // ========================================================

    atomicAdd(
        hard_overflow,
        1ULL
    );
}


/*
Diffuse particles currently stored in PRIMARY slots.
*/
extern "C" __global__
void diffuse_primary(

    const unsigned short* primary_in,
    const unsigned int* primary_occ_in,

    unsigned short* primary_out,
    unsigned int* primary_occ_out,

    unsigned short* spill_out,
    unsigned int* spill_occ_out,

    const int* neighbors,

    const float* direction_probability,

    unsigned long long* spill_writes,
    unsigned long long* rejected_moves,
    unsigned long long* hard_overflow,

    int primary_slots,
    int spill_slots,

    long long nvox,

    unsigned long long seed,
    unsigned long long step
)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    long long total =
        nvox
        * (long long)primary_slots;

    if (idx >= total)
    {
        return;
    }


    int voxel =
        idx / primary_slots;

    int slot =
        idx % primary_slots;


    if (
        slot >= primary_occ_in[voxel]
    )
    {
        return;
    }


    unsigned short species =
        primary_in[idx];

    if (species == 0)
    {
        return;
    }


    float q =
        direction_probability[
            species
        ];

    float total_hop_probability =
        6.0f * q;


    unsigned long long key =
        seed
        ^
        (
            (unsigned long long)idx
            * 0x9e3779b97f4a7c15ULL
        )
        ^
        (
            step
            * 0xbf58476d1ce4e5b9ULL
        );


    float r =
        random01(key);

    int destination =
        voxel;


    if (
        q > 0.0f
        &&
        r < total_hop_probability
    )
    {
        int direction =
            (int)(
                r / q
            );

        if (direction > 5)
        {
            direction = 5;
        }

        destination =
            neighbors[
                voxel * 6
                + direction
            ];
    }


    place_particle(
        species,

        destination,
        voxel,

        primary_out,
        primary_occ_out,

        spill_out,
        spill_occ_out,

        spill_writes,
        rejected_moves,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}


/*
Diffuse particles currently stored in SPILL slots.
*/
extern "C" __global__
void diffuse_spill(

    const unsigned short* spill_in,
    const unsigned int* spill_occ_in,

    unsigned short* primary_out,
    unsigned int* primary_occ_out,

    unsigned short* spill_out,
    unsigned int* spill_occ_out,

    const int* neighbors,

    const float* direction_probability,

    unsigned long long* spill_writes,
    unsigned long long* rejected_moves,
    unsigned long long* hard_overflow,

    int primary_slots,
    int spill_slots,

    long long nvox,

    unsigned long long seed,
    unsigned long long step
)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    long long total =
        nvox
        * (long long)spill_slots;

    if (idx >= total)
    {
        return;
    }


    int voxel =
        idx / spill_slots;

    int slot =
        idx % spill_slots;


    if (
        slot >= spill_occ_in[voxel]
    )
    {
        return;
    }


    unsigned short species =
        spill_in[idx];

    if (species == 0)
    {
        return;
    }


    float q =
        direction_probability[
            species
        ];

    float total_hop_probability =
        6.0f * q;


    unsigned long long key =
        seed
        ^
        (
            (unsigned long long)idx
            * 0xd6e8feb86659fd93ULL
        )
        ^
        (
            step
            * 0xa5a35625aae3e29dULL
        );


    float r =
        random01(key);

    int destination =
        voxel;


    if (
        q > 0.0f
        &&
        r < total_hop_probability
    )
    {
        int direction =
            (int)(
                r / q
            );

        if (direction > 5)
        {
            direction = 5;
        }

        destination =
            neighbors[
                voxel * 6
                + direction
            ];
    }


    place_particle(
        species,

        destination,
        voxel,

        primary_out,
        primary_occ_out,

        spill_out,
        spill_occ_out,

        spill_writes,
        rejected_moves,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}
