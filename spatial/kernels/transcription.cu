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


__device__ __forceinline__
int reserve_complex_slot(

    unsigned int* complex_active,

    int max_complexes,

    unsigned long long key
)
{
    if (max_complexes <= 0)
    {
        return -1;
    }


    int start =
        (int)(
            key
            % (unsigned long long)max_complexes
        );


    for (
        int offset = 0;
        offset < max_complexes;
        offset++
    )
    {
        int slot =
            start
            + offset;


        if (slot >= max_complexes)
        {
            slot -= max_complexes;
        }


        if (
            atomicCAS(
                &complex_active[
                    slot
                ],
                0U,
                1U
            )
            == 0U
        )
        {
            return slot;
        }
    }


    return -1;
}


__device__ __forceinline__
int reserve_mrna_slot(

    unsigned int* mrna_active,

    int max_mrna,

    unsigned long long key
)
{
    if (max_mrna <= 0)
    {
        return -1;
    }


    int start =
        (int)(
            key
            % (unsigned long long)max_mrna
        );


    for (
        int offset = 0;
        offset < max_mrna;
        offset++
    )
    {
        int slot =
            start
            + offset;


        if (slot >= max_mrna)
        {
            slot -= max_mrna;
        }


        // 2 = temporarily reserved

        if (
            atomicCAS(
                &mrna_active[
                    slot
                ],
                0U,
                2U
            )
            == 0U
        )
        {
            return slot;
        }
    }


    return -1;
}


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


        if (old >= capacity)
        {
            return false;
        }


        unsigned int previous =
            atomicCAS(
                &occupancy[
                    voxel
                ],
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


extern "C" __global__
void elongate_transcription(

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

    unsigned int* complex_active,
    int* complex_template,
    float* complex_remaining_bp,

    const int* completion_voxel,
    const unsigned short* rnap_species,
    const float* elongation_bp_per_s,

    unsigned int* mrna_active,
    int* mrna_template,
    int* mrna_voxel,

    int max_mrna,

    unsigned long long* completed_transcripts,
    unsigned long long* event_counts,

    int primary_slots,
    int spill_slots,

    int max_complexes,

    float dt,

    unsigned long long step
)
{
    int complex_index =
        blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        complex_index
        >= max_complexes
    )
    {
        return;
    }


    if (
        complex_active[
            complex_index
        ]
        == 0U
    )
    {
        return;
    }


    int template_index =
        complex_template[
            complex_index
        ];


    float remaining =
        complex_remaining_bp[
            complex_index
        ];


    remaining -= (
        elongation_bp_per_s[
            template_index
        ]
        * dt
    );


    if (remaining > 0.0f)
    {
        complex_remaining_bp[
            complex_index
        ] = remaining;

        return;
    }


    int voxel =
        completion_voxel[
            template_index
        ];


    unsigned short rnap =
        rnap_species[
            template_index
        ];


    unsigned long long key =
        (
            (unsigned long long)complex_index
            * 0x9e3779b97f4a7c15ULL
        )
        ^
        (
            step
            * 0xbf58476d1ce4e5b9ULL
        );


    // ========================================================
    // First reserve a slot for the new mRNA.
    //
    // If the mRNA pool is full, leave the transcription
    // complex complete-but-active and retry next step.
    // ========================================================

    int mrna_slot =
        reserve_mrna_slot(
            mrna_active,
            max_mrna,
            key
        );


    if (mrna_slot < 0)
    {
        complex_remaining_bp[
            complex_index
        ] = 0.0f;


        atomicAdd(
            &event_counts[
                4
            ],
            1ULL
        );


        return;
    }


    unsigned int particle_slot;


    // ========================================================
    // Return RNAP to the SAME 3' completion voxel.
    // ========================================================

    if (
        reserve_particle_slot(
            primary_occ,
            voxel,
            primary_slots,
            &particle_slot
        )
    )
    {
        primary[
            (long long)voxel
            * primary_slots
            + particle_slot
        ] = rnap;
    }


    else if (
        reserve_particle_slot(
            spill_occ,
            voxel,
            spill_slots,
            &particle_slot
        )
    )
    {
        spill[
            (long long)voxel
            * spill_slots
            + particle_slot
        ] = rnap;
    }


    else
    {
        // Give the reserved mRNA slot back.

        mrna_active[
            mrna_slot
        ] = 0U;


        complex_remaining_bp[
            complex_index
        ] = 0.0f;


        atomicAdd(
            &event_counts[
                2
            ],
            1ULL
        );


        return;
    }


    // ========================================================
    // Create the mRNA.
    //
    // Critically:
    //
    //     mrna_voxel = completion voxel
    //
    // so every transcript is born at its gene's 3' end.
    // ========================================================

    mrna_template[
        mrna_slot
    ] = template_index;


    mrna_voxel[
        mrna_slot
    ] = voxel;


    __threadfence();


    mrna_active[
        mrna_slot
    ] = 1U;


    atomicAdd(
        &completed_transcripts[
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


    complex_remaining_bp[
        complex_index
    ] = 0.0f;


    complex_template[
        complex_index
    ] = -1;


    __threadfence();


    complex_active[
        complex_index
    ] = 0U;
}


extern "C" __global__
void initiate_transcription(

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

    const int* voxel_offsets,
    const int* template_indices,

    const float* association_rates,
    const float* transcript_length_bp,
    const unsigned short* rnap_species,

    const int* blocker_offsets,
    const int* blocker_site_indices,
    const unsigned char* tf_bound,

    unsigned int* complex_active,
    int* complex_template,
    float* complex_remaining_bp,

    unsigned long long* event_counts,

    int primary_slots,
    int spill_slots,

    long long nvox,

    int max_complexes,

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
        np
        + ns;


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
        int template_index =
            template_indices[
                position
            ];


        bool blocked =
            false;


        int blocker_start =
            blocker_offsets[
                template_index
            ];


        int blocker_end =
            blocker_offsets[
                template_index + 1
            ];


        for (
            int blocker_position = blocker_start;
            blocker_position < blocker_end;
            blocker_position++
        )
        {
            int site_index =
                blocker_site_indices[
                    blocker_position
                ];


            if (
                tf_bound[
                    site_index
                ]
                != 0
            )
            {
                blocked = true;

                break;
            }
        }


        if (blocked)
        {
            continue;
        }


        unsigned short rnap =
            rnap_species[
                template_index
            ];


        unsigned int rnap_count =
            count_species(
                primary,
                spill,

                voxel,
                n,

                rnap,

                primary_slots,
                spill_slots
            );


        if (rnap_count == 0)
        {
            continue;
        }


        float hazard =
            association_rates[
                template_index
            ]
            * rnap_count;


        float probability =
            1.0f
            - expf(
                -hazard
                * dt
            );


        unsigned long long key =
            seed
            ^
            (
                (unsigned long long)voxel
                * 0x9e3779b97f4a7c15ULL
            )
            ^
            (
                (unsigned long long)template_index
                * 0xbf58476d1ce4e5b9ULL
            )
            ^
            (
                step
                * 0x94d049bb133111ebULL
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


        int complex_slot =
            reserve_complex_slot(
                complex_active,
                max_complexes,
                key
            );


        if (complex_slot < 0)
        {
            atomicAdd(
                &event_counts[
                    3
                ],
                1ULL
            );


            continue;
        }


        int rnap_slot =
            find_species(
                primary,
                spill,

                voxel,
                n,

                rnap,

                primary_slots,
                spill_slots
            );


        if (rnap_slot < 0)
        {
            complex_active[
                complex_slot
            ] = 0U;

            continue;
        }


        n--;


        if (
            (unsigned int)rnap_slot
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
                rnap_slot,

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


        complex_template[
            complex_slot
        ] = template_index;


        complex_remaining_bp[
            complex_slot
        ] = transcript_length_bp[
            template_index
        ];


        atomicAdd(
            &event_counts[
                0
            ],
            1ULL
        );
    }


    primary_occ[
        voxel
    ] = np;


    spill_occ[
        voxel
    ] = ns;
}