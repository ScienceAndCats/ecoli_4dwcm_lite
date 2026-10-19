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


extern "C" __global__
void degrade_mrna(

    unsigned int* active,
    int* template_index,
    int* voxel_id,
    unsigned int* bound_ribosomes,

    const float* degradation_rates,

    unsigned long long* degraded_by_template,
    unsigned long long* total_events,

    int max_mrna,

    float dt,

    unsigned long long seed,
    unsigned long long step
)
{
    int index =
        blockIdx.x
        * blockDim.x
        + threadIdx.x;


    if (
        index >= max_mrna
    )
    {
        return;
    }


    if (
        active[
            index
        ]
        != 1U
    )
    {
        return;
    }


    /*
    Temporary simplified rule:

        actively translating mRNA is protected from
        degradation.

    Later we can replace this with explicit ribosome runoff
    or co-translational degradation.
    */

    if (
        bound_ribosomes[
            index
        ]
        != 0U
    )
    {
        return;
    }


    int template_id =
        template_index[
            index
        ];


    if (
        template_id < 0
    )
    {
        return;
    }


    float rate =
        degradation_rates[
            template_id
        ];


    if (
        rate <= 0.0f
    )
    {
        return;
    }


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
            (unsigned long long)index
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


    atomicAdd(
        &degraded_by_template[
            template_id
        ],

        1ULL
    );


    atomicAdd(
        total_events,
        1ULL
    );


    template_index[
        index
    ] = -1;


    voxel_id[
        index
    ] = -1;


    bound_ribosomes[
        index
    ] = 0U;


    __threadfence();


    active[
        index
    ] = 0U;
}