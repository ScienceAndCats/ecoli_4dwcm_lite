#define PRODUCT_PARTICLE 1
#define PRODUCT_VOXEL_COUNT 2


__device__ __forceinline__
bool reserve_particle_slot(

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
                old + 1U
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
bool increment_uint32_if_possible(
    unsigned int* value
)
{
    while (true)
    {
        unsigned int old =
            atomicAdd(
                value,
                0U
            );


        if (
            old == 0xFFFFFFFFU
        )
        {
            return false;
        }


        if (
            atomicCAS(
                value,
                old,
                old + 1U
            )
            == old
        )
        {
            return true;
        }
    }
}


extern "C" __global__
void produce_proteins(

    unsigned int* product_event_active,
    int* product_event_template_index,
    int* product_event_voxel_id,

    const unsigned char* product_representation_codes,
    const int* product_numeric_ids,

    unsigned short* primary,
    unsigned int* primary_occupancy,

    unsigned short* spill,
    unsigned int* spill_occupancy,

    unsigned int* voxel_counts,

    unsigned long long* produced_by_template,
    unsigned long long* event_counts,

    int max_events,
    int cell_voxels,

    int primary_slots,
    int spill_slots,

    int voxel_count_species
)
{
    int event_index =
        blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        event_index >= max_events
    )
    {
        return;
    }


    if (
        product_event_active[
            event_index
        ]
        != 1U
    )
    {
        return;
    }


    int template_index =
        product_event_template_index[
            event_index
        ];


    int voxel =
        product_event_voxel_id[
            event_index
        ];


    if (
        template_index < 0
        ||
        voxel < 0
        ||
        voxel >= cell_voxels
    )
    {
        return;
    }


    unsigned char representation =
        product_representation_codes[
            template_index
        ];


    int product_id =
        product_numeric_ids[
            template_index
        ];


    bool produced =
        false;


    # ========================================================
    # Explicit particle product
    # ========================================================

    if (
        representation
        == PRODUCT_PARTICLE
    )
    {
        unsigned int slot;


        if (
            reserve_particle_slot(
                primary_occupancy,
                voxel,
                primary_slots,
                &slot
            )
        )
        {
            primary[
                (
                    (long long)voxel
                    * primary_slots
                )
                + slot
            ] = (
                (unsigned short)product_id
            );


            produced =
                true;
        }


        else if (
            reserve_particle_slot(
                spill_occupancy,
                voxel,
                spill_slots,
                &slot
            )
        )
        {
            spill[
                (
                    (long long)voxel
                    * spill_slots
                )
                + slot
            ] = (
                (unsigned short)product_id
            );


            produced =
                true;
        }


        else
        {
            /*
            Artificial particle-storage capacity is full.

            Keep the product event alive so nothing is lost.
            */

            atomicAdd(
                &event_counts[
                    1
                ],

                1ULL
            );


            return;
        }
    }


    # ========================================================
    # Integer voxel-count product
    # ========================================================

    else if (
        representation
        == PRODUCT_VOXEL_COUNT
    )
    {
        if (
            product_id < 0
            ||
            product_id >= voxel_count_species
        )
        {
            return;
        }


        bool incremented =
            increment_uint32_if_possible(
                &voxel_counts[
                    (
                        (long long)product_id
                        * cell_voxels
                    )
                    + voxel
                ]
            );


        if (
            !incremented
        )
        {
            atomicAdd(
                &event_counts[
                    2
                ],

                1ULL
            );


            return;
        }


        produced =
            true;
    }


    if (
        !produced
    )
    {
        return;
    }


    # ========================================================
    # Record successful product creation
    # ========================================================

    atomicAdd(
        &produced_by_template[
            template_index
        ],

        1ULL
    );


    atomicAdd(
        &event_counts[
            0
        ],

        1ULL
    );


    # ========================================================
    # Consume completion event only AFTER the protein exists.
    # ========================================================

    product_event_template_index[
        event_index
    ] = -1;


    product_event_voxel_id[
        event_index
    ] = -1;


    __threadfence();


    product_event_active[
        event_index
    ] = 0U;
}
