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
unsigned short get_particle(

    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,
    unsigned int logical_slot,

    int primary_slots,
    int spill_slots
)
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
    int spill_slots
)
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


__device__ __forceinline__
unsigned int count_species(

    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,

    unsigned int n,

    unsigned short species,

    int primary_slots,
    int spill_slots
)
{
    unsigned int count = 0;


    for (
        unsigned int i = 0;
        i < n;
        i++
    )
    {
        if (
            get_particle(
                primary,
                spill,

                voxel,
                i,

                primary_slots,
                spill_slots
            )
            == species
        )
        {
            count++;
        }
    }


    return count;
}


__device__ __forceinline__
int find_species(

    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,

    unsigned int n,

    unsigned short species,

    int primary_slots,
    int spill_slots
)
{
    for (
        unsigned int i = 0;
        i < n;
        i++
    )
    {
        if (
            get_particle(
                primary,
                spill,

                voxel,
                i,

                primary_slots,
                spill_slots
            )
            == species
        )
        {
            return (int)i;
        }
    }


    return -1;
}


extern "C" __global__
void promoter_binding_step(

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

    const int* voxel_offsets,
    const int* site_indices,

    const unsigned short* tf_species,
    const float* association_rates,
    const float* dissociation_rates,

    unsigned char* bound,

    unsigned long long* event_counts,

    int primary_slots,
    int spill_slots,

    long long nvox,

    float dt,

    unsigned long long seed,
    unsigned long long step
)
{
    long long voxel =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (voxel >= nvox)
    {
        return;
    }


    unsigned int np =
        primary_occ[
            voxel
        ];

    unsigned int ns =
        spill_occ[
            voxel
        ];

    unsigned int n =
        np + ns;

    unsigned int capacity =
        primary_slots
        + spill_slots;


    unsigned long long binding_events =
        0ULL;

    unsigned long long unbinding_events =
        0ULL;


    int start =
        voxel_offsets[
            voxel
        ];

    int end =
        voxel_offsets[
            voxel + 1
        ];


    for (
        int position = start;
        position < end;
        position++
    )
    {
        int site =
            site_indices[
                position
            ];


        unsigned short tf =
            tf_species[
                site
            ];


        unsigned long long key =
            seed
            ^
            (
                (unsigned long long)voxel
                * 0x9e3779b97f4a7c15ULL
            )
            ^
            (
                (unsigned long long)site
                * 0xbf58476d1ce4e5b9ULL
            )
            ^
            (
                step
                * 0x94d049bb133111ebULL
            );


        // ====================================================
        // Bound promoter:
        // possibly release the TF back into this voxel.
        // ====================================================

        if (
            bound[
                site
            ]
            != 0
        )
        {
            if (n >= capacity)
            {
                continue;
            }


            float rate =
                dissociation_rates[
                    site
                ];


            float probability =
                1.0f
                - expf(
                    -rate
                    * dt
                );


            if (
                random01(
                    key
                )
                < probability
            )
            {
                set_particle(
                    primary,
                    spill,

                    voxel,
                    n,

                    tf,

                    primary_slots,
                    spill_slots
                );


                if (
                    n
                    <
                    (unsigned int)primary_slots
                )
                {
                    np++;
                }

                else
                {
                    ns++;
                }


                n++;

                bound[
                    site
                ] = 0;

                unbinding_events++;
            }


            continue;
        }


        // ====================================================
        // Unbound promoter:
        // determine how many matching free TF molecules are
        // currently in this promoter's voxel.
        // ====================================================

        unsigned int tf_count =
            count_species(
                primary,
                spill,

                voxel,

                n,

                tf,

                primary_slots,
                spill_slots
            );


        if (tf_count == 0)
        {
            continue;
        }


        float hazard =
            association_rates[
                site
            ]
            * tf_count;


        float probability =
            1.0f
            - expf(
                -hazard
                * dt
            );


        if (
            random01(
                key
            )
            >= probability
        )
        {
            continue;
        }


        int tf_slot =
            find_species(
                primary,
                spill,

                voxel,

                n,

                tf,

                primary_slots,
                spill_slots
            );


        if (tf_slot < 0)
        {
            continue;
        }


        // ====================================================
        // Remove the free TF particle from the voxel.
        //
        // The TF now exists as promoter-bound state instead.
        // ====================================================

        n--;


        if (
            (unsigned int)tf_slot
            != n
        )
        {
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
                tf_slot,

                last,

                primary_slots,
                spill_slots
            );
        }


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
        {
            ns--;
        }

        else
        {
            np--;
        }


        bound[
            site
        ] = 1;

        binding_events++;
    }


    primary_occ[
        voxel
    ] = np;

    spill_occ[
        voxel
    ] = ns;


    if (binding_events > 0)
    {
        atomicAdd(
            &event_counts[0],
            binding_events
        );
    }


    if (unbinding_events > 0)
    {
        atomicAdd(
            &event_counts[1],
            unbinding_events
        );
    }
}
