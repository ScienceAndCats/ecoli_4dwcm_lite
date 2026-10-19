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
bool decrement_if_positive(
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
            old == 0U
        )
        {
            return false;
        }


        if (
            atomicCAS(
                value,
                old,
                old - 1U
            )
            == old
        )
        {
            return true;
        }
    }
}


__device__ __forceinline__
int reserve_slot(

    unsigned int* active,

    int maximum,

    unsigned int start
)
{
    for (
        int offset = 0;
        offset < maximum;
        offset++
    )
    {
        int index =
            (
                start
                + offset
            )
            % maximum;


        if (
            atomicCAS(
                &active[
                    index
                ],

                0U,
                2U
            )
            == 0U
        )
        {
            return index;
        }
    }


    return -1;
}


extern "C" __global__
void elongate_translation(

    unsigned int* complex_active,

    int* complex_mrna_index,
    int* complex_template_index,
    int* complex_voxel_id,

    float* complex_remaining_aa,

    const float* elongation_rates,

    unsigned int* free_ribosomes,

    unsigned int* mrna_bound_ribosomes,

    unsigned int* product_event_active,
    int* product_event_template_index,
    int* product_event_voxel_id,

    unsigned long long* completed_by_template,
    unsigned long long* event_counts,

    int max_complexes,

    float dt
)
{
    int index =
        blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        index >= max_complexes
    )
    {
        return;
    }


    if (
        complex_active[
            index
        ]
        != 1U
    )
    {
        return;
    }


    int mrna_index =
        complex_mrna_index[
            index
        ];


    int template_index =
        complex_template_index[
            index
        ];


    int voxel =
        complex_voxel_id[
            index
        ];


    if (
        mrna_index < 0
        ||
        template_index < 0
        ||
        voxel < 0
    )
    {
        return;
    }


    float remaining =
        complex_remaining_aa[
            index
        ];


    remaining -= (
        elongation_rates[
            template_index
        ]
        * dt
    );


    if (
        remaining > 0.0f
    )
    {
        complex_remaining_aa[
            index
        ] = remaining;

        return;
    }


    /*
    Translation is chemically complete, but before releasing
    the ribosome we MUST reserve a product-event slot.

    If there is nowhere to record the finished protein, the
    translation complex stays active and retries next step.
    */


    unsigned int event_start =
        (
            (unsigned int)index
        )
        % (
            (unsigned int)max_complexes
        );


    int product_event =
        reserve_slot(
            product_event_active,
            max_complexes,
            event_start
        );


    if (
        product_event < 0
    )
    {
        complex_remaining_aa[
            index
        ] = 0.0f;


        atomicAdd(
            &event_counts[
                4
            ],

            1ULL
        );


        return;
    }


    product_event_template_index[
        product_event
    ] = template_index;


    product_event_voxel_id[
        product_event
    ] = voxel;


    __threadfence();


    product_event_active[
        product_event
    ] = 1U;


    /*
    The protein completion event is now safely recorded.
    The ribosome may be released.
    */


    atomicAdd(
        &free_ribosomes[
            voxel
        ],

        1U
    );


    decrement_if_positive(
        &mrna_bound_ribosomes[
            mrna_index
        ]
    );


    atomicAdd(
        &completed_by_template[
            template_index
        ],

        1ULL
    );


    atomicAdd(
        &event_counts[
            1
        ],

        1ULL
    );


    complex_mrna_index[
        index
    ] = -1;


    complex_template_index[
        index
    ] = -1;


    complex_voxel_id[
        index
    ] = -1;


    complex_remaining_aa[
        index
    ] = 0.0f;


    __threadfence();


    complex_active[
        index
    ] = 0U;
}


extern "C" __global__
void initiate_translation(

    const unsigned int* mrna_active,
    const int* mrna_template_index,
    const int* mrna_voxel_id,

    unsigned int* mrna_bound_ribosomes,

    unsigned int* free_ribosomes,

    const unsigned char* template_enabled,

    const float* association_rates,

    const float* protein_lengths_aa,

    const unsigned int* max_ribosomes_per_mrna,

    unsigned int* complex_active,

    int* complex_mrna_index,
    int* complex_template_index,
    int* complex_voxel_id,

    float* complex_remaining_aa,

    unsigned long long* initiated_by_template,
    unsigned long long* event_counts,

    int max_mrna,
    int max_complexes,

    float dt,

    unsigned long long seed,
    unsigned long long step
)
{
    int mrna_index =
        blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        mrna_index >= max_mrna
    )
    {
        return;
    }


    if (
        mrna_active[
            mrna_index
        ]
        != 1U
    )
    {
        return;
    }


    int template_index =
        mrna_template_index[
            mrna_index
        ];


    if (
        template_index < 0
    )
    {
        return;
    }


    if (
        template_enabled[
            template_index
        ]
        == 0U
    )
    {
        return;
    }


    int voxel =
        mrna_voxel_id[
            mrna_index
        ];


    if (
        voxel < 0
    )
    {
        return;
    }


    unsigned int bound =
        mrna_bound_ribosomes[
            mrna_index
        ];


    unsigned int maximum_bound =
        max_ribosomes_per_mrna[
            template_index
        ];


    if (
        bound >= maximum_bound
    )
    {
        atomicAdd(
            &event_counts[
                3
            ],

            1ULL
        );

        return;
    }


    unsigned int free_count =
        atomicAdd(
            &free_ribosomes[
                voxel
            ],

            0U
        );


    if (
        free_count == 0U
    )
    {
        return;
    }


    float rate =
        association_rates[
            template_index
        ]
        * (float)free_count;


    float probability =
        1.0f
        - expf(
            -rate
            * dt
        );


    unsigned long long key =
        seed
        ^
        (
            (unsigned long long)mrna_index
            * 0x9e3779b97f4a7c15ULL
        )
        ^
        (
            step
            * 0xbf58476d1ce4e5b9ULL
        );


    if (
        random01(
            key
        )
        >= probability
    )
    {
        return;
    }


    unsigned int start =
        (
            (unsigned int)mrna_index
            +
            (unsigned int)step
        )
        % (
            (unsigned int)max_complexes
        );


    int complex_index =
        reserve_slot(
            complex_active,
            max_complexes,
            start
        );


    if (
        complex_index < 0
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


    bool acquired_ribosome =
        decrement_if_positive(
            &free_ribosomes[
                voxel
            ]
        );


    if (
        !acquired_ribosome
    )
    {
        complex_active[
            complex_index
        ] = 0U;

        return;
    }


    atomicAdd(
        &mrna_bound_ribosomes[
            mrna_index
        ],

        1U
    );


    complex_mrna_index[
        complex_index
    ] = mrna_index;


    complex_template_index[
        complex_index
    ] = template_index;


    complex_voxel_id[
        complex_index
    ] = voxel;


    complex_remaining_aa[
        complex_index
    ] = protein_lengths_aa[
        template_index
    ];


    atomicAdd(
        &initiated_by_template[
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


    __threadfence();


    complex_active[
        complex_index
    ] = 1U;
}