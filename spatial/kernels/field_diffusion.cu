extern "C" __global__
void diffuse_fields(

    const float* input,
    float* output,

    const int* neighbors,

    const float* alpha_by_field,

    long long nvox,
    long long nfields,
    long long total
)
{
    long long idx =
        (long long)blockIdx.x
        * blockDim.x
        + threadIdx.x;

    if (idx >= total)
    {
        return;
    }


    long long field =
        idx / nvox;

    long long voxel =
        idx % nvox;


    float center =
        input[idx];

    float alpha =
        alpha_by_field[field];


    float laplacian =
        0.0f;


    for (
        int direction = 0;
        direction < 6;
        direction++
    )
    {
        int neighbor =
            neighbors[
                voxel * 6
                + direction
            ];

        long long neighbor_idx =
            field * nvox
            + neighbor;

        laplacian += (
            input[neighbor_idx]
            - center
        );
    }


    output[idx] =
        center
        + alpha * laplacian;
}
