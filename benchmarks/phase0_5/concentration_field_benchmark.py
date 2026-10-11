import argparse
import time
import numpy as np
import cupy as cp


parser = argparse.ArgumentParser()
parser.add_argument("--fields", type=int, required=True)
parser.add_argument("--diffusion-steps", type=int, default=50)
parser.add_argument("--reaction-steps", type=int, default=100)
args = parser.parse_args()


# Same 50-nm bounding lattice used in our LM benchmark
NX = 64
NY = 32
NZ = 32

NVOX = NX * NY * NZ
NFIELDS = args.fields
NVALUES = NFIELDS * NVOX

print(f"Fields: {NFIELDS:,}")
print(f"Voxels: {NVOX:,}")
print(f"Concentration values: {NVALUES:,}")


# ---------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------

diffusion_kernel = cp.RawKernel(r'''
extern "C" __global__
void diffuse(
    const float* in,
    float* out,
    const int nx,
    const int ny,
    const int nz,
    const long long nvox,
    const long long total,
    const float alpha)
{
    long long idx =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total)
        return;

    long long v = idx % nvox;

    int x = v % nx;
    int y = (v / nx) % ny;
    int z = v / ((long long)nx * ny);

    float c = in[idx];

    float xm = (x > 0)      ? in[idx - 1]       : c;
    float xp = (x < nx - 1) ? in[idx + 1]       : c;

    float ym = (y > 0)      ? in[idx - nx]      : c;
    float yp = (y < ny - 1) ? in[idx + nx]      : c;

    long long plane = (long long)nx * ny;

    float zm = (z > 0)      ? in[idx - plane]   : c;
    float zp = (z < nz - 1) ? in[idx + plane]   : c;

    out[idx] =
        c + alpha * (xm + xp + ym + yp + zm + zp - 6.0f*c);
}
''', 'diffuse')


reaction_kernel = cp.RawKernel(r'''
extern "C" __global__
void react(
    const float* in,
    float* out,
    const float* factors,
    const long long nvox,
    const long long total)
{
    long long idx =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= total)
        return;

    long long species = idx / nvox;

    out[idx] = in[idx] * factors[species];
}
''', 'react')


# ---------------------------------------------------------
# Allocate concentration fields
# ---------------------------------------------------------

t0 = time.perf_counter()

a = cp.ones(NVALUES, dtype=cp.float32)
b = cp.empty_like(a)

# Each field gets a slightly different local reaction rate
factors = cp.linspace(
    0.9990,
    0.9999,
    NFIELDS,
    dtype=cp.float32
)

cp.cuda.Stream.null.synchronize()

allocation_seconds = time.perf_counter() - t0


threads = 256
blocks = (NVALUES + threads - 1) // threads


# ---------------------------------------------------------
# Warm up kernels / compile CUDA
# ---------------------------------------------------------

diffusion_kernel(
    (blocks,),
    (threads,),
    (
        a, b,
        np.int32(NX),
        np.int32(NY),
        np.int32(NZ),
        np.int64(NVOX),
        np.int64(NVALUES),
        np.float32(0.1)
    )
)

reaction_kernel(
    (blocks,),
    (threads,),
    (
        a, b,
        factors,
        np.int64(NVOX),
        np.int64(NVALUES)
    )
)

cp.cuda.Stream.null.synchronize()


# ---------------------------------------------------------
# Diffusion benchmark
# ---------------------------------------------------------

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

for _ in range(args.diffusion_steps):

    diffusion_kernel(
        (blocks,),
        (threads,),
        (
            a, b,
            np.int32(NX),
            np.int32(NY),
            np.int32(NZ),
            np.int64(NVOX),
            np.int64(NVALUES),
            np.float32(0.1)
        )
    )

    a, b = b, a

end.record()
end.synchronize()

diffusion_ms = cp.cuda.get_elapsed_time(start, end)


# ---------------------------------------------------------
# Local reaction benchmark
# ---------------------------------------------------------

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

for _ in range(args.reaction_steps):

    reaction_kernel(
        (blocks,),
        (threads,),
        (
            a, b,
            factors,
            np.int64(NVOX),
            np.int64(NVALUES)
        )
    )

    a, b = b, a

end.record()
end.synchronize()

reaction_ms = cp.cuda.get_elapsed_time(start, end)


# ---------------------------------------------------------
# Memory
# ---------------------------------------------------------

pool = cp.get_default_memory_pool()

used_mib = pool.used_bytes() / 1024**2
reserved_mib = pool.total_bytes() / 1024**2


print()
print("RESULTS")
print("---------------------------------------")
print(f"Allocation time:       {allocation_seconds:.3f} s")
print(f"GPU memory in use:     {used_mib:.1f} MiB")
print(f"GPU memory reserved:   {reserved_mib:.1f} MiB")

print()
print(
    f"Diffusion:             "
    f"{args.diffusion_steps} updates in "
    f"{diffusion_ms/1000:.3f} s"
)

print(
    f"Diffusion/update:      "
    f"{diffusion_ms/args.diffusion_steps:.3f} ms"
)

print()
print(
    f"Local reactions:       "
    f"{args.reaction_steps} updates in "
    f"{reaction_ms/1000:.3f} s"
)

print(
    f"Reaction/update:       "
    f"{reaction_ms/args.reaction_steps:.3f} ms"
)

print()
print(
    f"Final checksum:        "
    f"{float(cp.sum(a)):.3e}"
)
