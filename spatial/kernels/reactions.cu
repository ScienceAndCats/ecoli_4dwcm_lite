#define REACTION_DECAY          1
#define REACTION_CONVERSION     2
#define REACTION_CATALYTIC      3
#define REACTION_BINDING        4
#define REACTION_UNBINDING      5
#define REACTION_PARTICLE_FIELD 6


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

    unsigned int exclude_slot,

    unsigned short species,

    int primary_slots,
    int spill_slots
)
{
    unsigned int count = 0;


    for (
        unsigned int j = 0;
        j < n;
        j++
    )
    {
        if (j == exclude_slot)
        {
            continue;
        }


        if (
            get_particle(
                primary,
                spill,

                voxel,
                j,

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
int find_partner(

    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,

    unsigned int n,

    unsigned int exclude_slot,

    unsigned short species,

    int primary_slots,
    int spill_slots
)
{
    for (
        unsigned int j = 0;
        j < n;
        j++
    )
    {
        if (j == exclude_slot)
        {
            continue;
        }


        if (
            get_particle(
                primary,
                spill,

                voxel,
                j,

                primary_slots,
                spill_slots
            )
            == species
        )
        {
            return (int)j;
        }
    }


    return -1;
}


__device__ __forceinline__
float reaction_hazard(

    unsigned char type,

    unsigned short partner,

    float rate,

    const unsigned short* primary,
    const unsigned short* spill,

    long long voxel,

    unsigned int current_slot,
    unsigned int n,
    unsigned int capacity,

    int primary_slots,
    int spill_slots
)
{
    if (
        type == REACTION_DECAY
        ||
        type == REACTION_CONVERSION
    )
    {
        return rate;
    }


    if (
        type == REACTION_CATALYTIC
        ||
        type == REACTION_BINDING
    )
    {
        unsigned int partner_count =
            count_species(
                primary,
                spill,

                voxel,

                n,
                current_slot,

                partner,

                primary_slots,
                spill_slots
            );


        return (
            rate
            * partner_count
        );
    }


    if (
        type == REACTION_UNBINDING
    )
    {
        if (n >= capacity)
        {
            return 0.0f;
        }

        return rate;
    }


    return 0.0f;
}


extern "C" __global__
void reaction_step(

    unsigned short* primary,
    unsigned int* primary_occ,

    unsigned short* spill,
    unsigned int* spill_occ,

    float* fields,

    const int* reaction_offsets,

    const unsigned char* reaction_types,

    const unsigned short* partner_species,
    const unsigned short* product_a,
    const unsigned short* product_b,

    const float* rates,

    const int* field_substrate,
    const int* field_product,

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


    unsigned long long local_events[6] = {
        0ULL,
        0ULL,
        0ULL,
        0ULL,
        0ULL,
        0ULL
    };


    unsigned int i = 0;


    while (i < n)
    {
        unsigned short species =
            get_particle(
                primary,
                spill,

                voxel,
                i,

                primary_slots,
                spill_slots
            );


        if (species == 0)
        {
            i++;
            continue;
        }


        int start =
            reaction_offsets[
                species
            ];

        int end =
            reaction_offsets[
                species + 1
            ];


        if (start == end)
        {
            i++;
            continue;
        }


        // ====================================================
        // Continuous particle-field reactions
        // ====================================================

        for (
            int r = start;
            r < end;
            r++
        )
        {
            if (
                reaction_types[r]
                != REACTION_PARTICLE_FIELD
            )
            {
                continue;
            }


            int fs =
                field_substrate[r];

            int fp =
                field_product[r];


            if (
                fs < 0
                ||
                fp < 0
            )
            {
                continue;
            }


            long long substrate_index =
                (long long)fs
                * nvox
                + voxel;

            long long product_index =
                (long long)fp
                * nvox
                + voxel;


            float substrate =
                fields[
                    substrate_index
                ];


            if (substrate <= 0.0f)
            {
                continue;
            }


            float flux =
                rates[r]
                * substrate
                * dt;


            if (flux > substrate)
            {
                flux = substrate;
            }


            fields[
                substrate_index
            ] -= flux;

            fields[
                product_index
            ] += flux;


            local_events[5]++;
        }


        // ====================================================
        // Calculate total hazard for all discrete reactions
        // available to this particle.
        // ====================================================

        float total_hazard =
            0.0f;


        for (
            int r = start;
            r < end;
            r++
        )
        {
            unsigned char type =
                reaction_types[r];


            if (
                type
                == REACTION_PARTICLE_FIELD
            )
            {
                continue;
            }


            total_hazard +=
                reaction_hazard(
                    type,

                    partner_species[r],

                    rates[r],

                    primary,
                    spill,

                    voxel,

                    i,
                    n,
                    capacity,

                    primary_slots,
                    spill_slots
                );
        }


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
            seed
            ^
            (
                (unsigned long long)voxel
                * 0x9e3779b97f4a7c15ULL
            )
            ^
            (
                (unsigned long long)i
                * 0xbf58476d1ce4e5b9ULL
            )
            ^
            (
                step
                * 0x94d049bb133111ebULL
            );


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


        int selected_reaction =
            -1;


        for (
            int r = start;
            r < end;
            r++
        )
        {
            unsigned char type =
                reaction_types[r];


            if (
                type
                == REACTION_PARTICLE_FIELD
            )
            {
                continue;
            }


            float hazard =
                reaction_hazard(
                    type,

                    partner_species[r],

                    rates[r],

                    primary,
                    spill,

                    voxel,

                    i,
                    n,
                    capacity,

                    primary_slots,
                    spill_slots
                );


            if (chooser < hazard)
            {
                selected_reaction = r;
                break;
            }


            chooser -= hazard;
        }


        if (selected_reaction < 0)
        {
            i++;
            continue;
        }


        unsigned char selected_type =
            reaction_types[
                selected_reaction
            ];


        // ====================================================
        // DECAY
        // ====================================================

        if (
            selected_type
            == REACTION_DECAY
        )
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
            {
                ns--;
            }
            else
            {
                np--;
            }


            local_events[0]++;

            continue;
        }


        // ====================================================
        // CONVERSION
        // ====================================================

        if (
            selected_type
            == REACTION_CONVERSION
        )
        {
            set_particle(
                primary,
                spill,

                voxel,
                i,

                product_a[
                    selected_reaction
                ],

                primary_slots,
                spill_slots
            );


            local_events[1]++;

            i++;

            continue;
        }


        // ====================================================
        // CATALYTIC CONVERSION
        // ====================================================

        if (
            selected_type
            == REACTION_CATALYTIC
        )
        {
            set_particle(
                primary,
                spill,

                voxel,
                i,

                product_a[
                    selected_reaction
                ],

                primary_slots,
                spill_slots
            );


            local_events[2]++;

            i++;

            continue;
        }


        // ====================================================
        // BINDING
        // ====================================================

        if (
            selected_type
            == REACTION_BINDING
        )
        {
            int partner_slot =
                find_partner(
                    primary,
                    spill,

                    voxel,

                    n,
                    i,

                    partner_species[
                        selected_reaction
                    ],

                    primary_slots,
                    spill_slots
                );


            if (partner_slot < 0)
            {
                i++;
                continue;
            }


            // Current particle becomes the complex.

            set_particle(
                primary,
                spill,

                voxel,
                i,

                product_a[
                    selected_reaction
                ],

                primary_slots,
                spill_slots
            );


            // Remove the partner by replacing it with
            // the final logical particle.

            n--;


            if (
                (unsigned int)partner_slot
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
                    partner_slot,

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


            local_events[3]++;

            i++;

            continue;
        }


        // ====================================================
        // UNBINDING
        // ====================================================

        if (
            selected_type
            == REACTION_UNBINDING
        )
        {
            if (n >= capacity)
            {
                i++;
                continue;
            }


            set_particle(
                primary,
                spill,

                voxel,
                i,

                product_a[
                    selected_reaction
                ],

                primary_slots,
                spill_slots
            );


            set_particle(
                primary,
                spill,

                voxel,
                n,

                product_b[
                    selected_reaction
                ],

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

            local_events[4]++;

            i++;

            continue;
        }


        i++;
    }


    primary_occ[
        voxel
    ] = np;

    spill_occ[
        voxel
    ] = ns;


    for (
        int e = 0;
        e < 6;
        e++
    )
    {
        event_counts[
            voxel * 6
            + e
        ] += local_events[e];
    }
}
