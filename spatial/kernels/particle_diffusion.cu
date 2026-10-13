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
                &occupancy[
                    voxel
                ],
                0U
            );


        if (
            old >= capacity
        )
        {
            return false;
        }


        if (
            atomicCAS(
                &occupancy[
                    voxel
                ],
                old,
                old + 1
            )
            == old
        )
        {
            *slot_out = old;

            return true;
        }
    }
}


__device__ __forceinline__
bool reserve_physical_volume(

    int destination,

    float molecule_volume_nm3,

    const float* occupied_volume_nm3,
    float* reserved_incoming_nm3,

    float voxel_volume_nm3
)
{
    if (
        molecule_volume_nm3
        <= 0.0f
    )
    {
        return true;
    }


    float previous_reserved =
        atomicAdd(
            &reserved_incoming_nm3[
                destination
            ],

            molecule_volume_nm3
        );


    float resulting_volume =
        occupied_volume_nm3[
            destination
        ]
        + previous_reserved
        + molecule_volume_nm3;


    if (
        resulting_volume
        <= voxel_volume_nm3
    )
    {
        return true;
    }


    atomicAdd(
        &reserved_incoming_nm3[
            destination
        ],

        -molecule_volume_nm3
    );


    return false;
}


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
    unsigned long long* slot_rejected_moves,
    unsigned long long* hard_overflow,

    int primary_slots,
    int spill_slots
)
{
    unsigned int slot;


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


    if (
        destination_voxel
        != source_voxel
    )
    {
        atomicAdd(
            slot_rejected_moves,
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


    atomicAdd(
        hard_overflow,
        1ULL
    );
}


/*
Apply:

    1. soft crowding accessibility
    2. hard physical-volume exclusion

Returns the actual destination.
*/
__device__ __forceinline__
int evaluate_destination(

    int source_voxel,
    int proposed_destination,

    unsigned short species,

    const float* movement_accessibility,

    const float* species_volume_nm3,

    const float* occupied_volume_nm3,

    float* reserved_incoming_nm3,

    float voxel_volume_nm3,

    int use_crowding,
    int use_capacity,

    unsigned long long key,

    unsigned long long* crowding_rejections,
    unsigned long long* capacity_rejections
)
{
    if (
        proposed_destination
        == source_voxel
    )
    {
        return source_voxel;
    }


    #pragma unroll
    for (
        int dummy = 0;
        dummy < 1;
        dummy++
    )
    {
        if (
            use_crowding
            != 0
        )
        {
            float accessibility =
                movement_accessibility[
                    proposed_destination
                ];


            float r =
                random01(
                    key
                    ^ 0x4f1bbcdc67657a35ULL
                );


            if (
                r
                >= accessibility
            )
            {
                atomicAdd(
                    crowding_rejections,
                    1ULL
                );

                return source_voxel;
            }
        }


        if (
            use_capacity
            != 0
        )
        {
            float molecule_volume =
                species_volume_nm3[
                    species
                ];


            bool fits =
                reserve_physical_volume(
                    proposed_destination,

                    molecule_volume,

                    occupied_volume_nm3,

                    reserved_incoming_nm3,

                    voxel_volume_nm3
                );


            if (
                !fits
            )
            {
                atomicAdd(
                    capacity_rejections,
                    1ULL
                );

                return source_voxel;
            }
        }
    }


    return proposed_destination;
}


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
    const float* movement_accessibility,

    const float* species_volume_nm3,

    const float* occupied_volume_nm3,

    float* reserved_incoming_nm3,

    unsigned long long* spill_writes,
    unsigned long long* slot_rejected_moves,
    unsigned long long* hard_overflow,
    unsigned long long* crowding_rejections,
    unsigned long long* capacity_rejections,

    int primary_slots,
    int spill_slots,

    long long nvox,

    float voxel_volume_nm3,

    int use_crowding,
    int use_capacity,

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


    if (
        idx >= total
    )
    {
        return;
    }


    int voxel =
        idx
        / primary_slots;


    int slot =
        idx
        % primary_slots;


    if (
        slot
        >= primary_occ_in[
            voxel
        ]
    )
    {
        return;
    }


    unsigned short species =
        primary_in[
            idx
        ];


    if (
        species == 0
    )
    {
        return;
    }


    float q =
        direction_probability[
            species
        ];


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
        random01(
            key
        );


    int destination =
        voxel;


    if (
        q > 0.0f
        &&
        r < 6.0f * q
    )
    {
        int direction =
            (int)(
                r / q
            );


        if (
            direction > 5
        )
        {
            direction = 5;
        }


        int proposed =
            neighbors[
                voxel * 6
                + direction
            ];


        destination =
            evaluate_destination(
                voxel,
                proposed,

                species,

                movement_accessibility,

                species_volume_nm3,

                occupied_volume_nm3,

                reserved_incoming_nm3,

                voxel_volume_nm3,

                use_crowding,
                use_capacity,

                key,

                crowding_rejections,
                capacity_rejections
            );
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
        slot_rejected_moves,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}


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
    const float* movement_accessibility,

    const float* species_volume_nm3,

    const float* occupied_volume_nm3,

    float* reserved_incoming_nm3,

    unsigned long long* spill_writes,
    unsigned long long* slot_rejected_moves,
    unsigned long long* hard_overflow,
    unsigned long long* crowding_rejections,
    unsigned long long* capacity_rejections,

    int primary_slots,
    int spill_slots,

    long long nvox,

    float voxel_volume_nm3,

    int use_crowding,
    int use_capacity,

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


    if (
        idx >= total
    )
    {
        return;
    }


    int voxel =
        idx
        / spill_slots;


    int slot =
        idx
        % spill_slots;


    if (
        slot
        >= spill_occ_in[
            voxel
        ]
    )
    {
        return;
    }


    unsigned short species =
        spill_in[
            idx
        ];


    if (
        species == 0
    )
    {
        return;
    }


    float q =
        direction_probability[
            species
        ];


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
        random01(
            key
        );


    int destination =
        voxel;


    if (
        q > 0.0f
        &&
        r < 6.0f * q
    )
    {
        int direction =
            (int)(
                r / q
            );


        if (
            direction > 5
        )
        {
            direction = 5;
        }


        int proposed =
            neighbors[
                voxel * 6
                + direction
            ];


        destination =
            evaluate_destination(
                voxel,
                proposed,

                species,

                movement_accessibility,

                species_volume_nm3,

                occupied_volume_nm3,

                reserved_incoming_nm3,

                voxel_volume_nm3,

                use_crowding,
                use_capacity,

                key,

                crowding_rejections,
                capacity_rejections
            );
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
        slot_rejected_moves,
        hard_overflow,

        primary_slots,
        spill_slots
    );
}