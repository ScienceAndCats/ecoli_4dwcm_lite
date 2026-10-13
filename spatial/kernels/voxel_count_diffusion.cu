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
Try to reserve enough physical volume in a destination voxel.

occupied_volume_nm3 is the occupancy snapshot at the start
of this diffusion step.

reserved_incoming_nm3 records molecules that have already
been accepted into that destination during THIS step.

This prevents many GPU threads from all independently seeing
the same available volume and collectively overfilling it.
*/
__device__ __forceinline__
bool reserve_physical_volume(

    int destination,

    float molecule_volume_nm3,

    const float* occupied_volume_nm3,
    float* reserved_incoming_nm3,

    float voxel_volume_nm3
)
{
    /*
    A zero-volume entry means no hard physical exclusion has
    been configured for this species.
    */

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


    /*
    Reservation did not fit.
    Undo it.
    */

    atomicAdd(
        &reserved_incoming_nm3[
            destination
        ],

        -molecule_volume_nm3
    );


    return false;
}


extern "C" __global__
void diffuse_voxel_counts(

    const unsigned int* counts_in,
    unsigned int* counts_out,

    const int* neighbors,

    const float* direction_probability,

    const float* movement_accessibility,

    const float* species_volume_nm3,

    const float* occupied_volume_nm3,

    float* reserved_incoming_nm3,

    unsigned long long* attempted_hops,
    unsigned long long* actual_hops,
    unsigned long long* crowding_rejections,
    unsigned long long* capacity_rejections,

    long long nvox,
    long long nspecies,
    long long total_entries,

    float voxel_volume_nm3,

    int use_crowding,
    int use_capacity,

    unsigned long long seed,
    unsigned long long step
)
{
    long long index =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        index >= total_entries
    )
    {
        return;
    }


    int species =
        (int)(
            index / nvox
        );


    int voxel =
        (int)(
            index
            - (
                (long long)species
                * nvox
            )
        );


    unsigned int count =
        counts_in[
            index
        ];


    if (
        count == 0U
    )
    {
        return;
    }


    float q =
        direction_probability[
            species
        ];


    float total_hop_probability =
        6.0f
        * q;


    float molecule_volume =
        species_volume_nm3[
            species
        ];


    unsigned long long local_attempted =
        0ULL;

    unsigned long long local_actual =
        0ULL;

    unsigned long long local_crowding_rejected =
        0ULL;

    unsigned long long local_capacity_rejected =
        0ULL;


    for (
        unsigned int molecule = 0;
        molecule < count;
        molecule++
    )
    {
        int destination =
            voxel;


        unsigned long long key =
            seed
            ^
            (
                (unsigned long long)index
                * 0x9e3779b97f4a7c15ULL
            )
            ^
            (
                (unsigned long long)molecule
                * 0xbf58476d1ce4e5b9ULL
            )
            ^
            (
                step
                * 0x94d049bb133111ebULL
            );


        float r =
            random01(
                key
            );


        if (
            q > 0.0f
            &&
            r < total_hop_probability
        )
        {
            local_attempted++;


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


            int proposed_destination =
                neighbors[
                    voxel * 6
                    + direction
                ];


            if (
                proposed_destination
                != voxel
            )
            {
                bool soft_accept =
                    true;


                /*
                First apply the soft crowding/accessibility
                probability.
                */

                if (
                    use_crowding
                    != 0
                )
                {
                    float accessibility =
                        movement_accessibility[
                            proposed_destination
                        ];


                    float access_random =
                        random01(
                            key
                            ^ 0xd6e8feb86659fd93ULL
                        );


                    if (
                        access_random
                        >= accessibility
                    )
                    {
                        soft_accept =
                            false;

                        local_crowding_rejected++;
                    }
                }


                if (
                    soft_accept
                )
                {
                    bool physical_fit =
                        true;


                    /*
                    Then apply hard physical exclusion.
                    */

                    if (
                        use_capacity
                        != 0
                    )
                    {
                        physical_fit =
                            reserve_physical_volume(
                                proposed_destination,

                                molecule_volume,

                                occupied_volume_nm3,

                                reserved_incoming_nm3,

                                voxel_volume_nm3
                            );
                    }


                    if (
                        physical_fit
                    )
                    {
                        destination =
                            proposed_destination;

                        local_actual++;
                    }

                    else
                    {
                        local_capacity_rejected++;
                    }
                }
            }
        }


        atomicAdd(
            &counts_out[
                (
                    (long long)species
                    * nvox
                )
                + destination
            ],

            1U
        );
    }


    if (
        local_attempted > 0ULL
    )
    {
        atomicAdd(
            attempted_hops,
            local_attempted
        );
    }


    if (
        local_actual > 0ULL
    )
    {
        atomicAdd(
            actual_hops,
            local_actual
        );
    }


    if (
        local_crowding_rejected > 0ULL
    )
    {
        atomicAdd(
            crowding_rejections,
            local_crowding_rejected
        );
    }


    if (
        local_capacity_rejected > 0ULL
    )
    {
        atomicAdd(
            capacity_rejections,
            local_capacity_rejected
        );
    }
}