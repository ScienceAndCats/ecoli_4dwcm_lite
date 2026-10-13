import cupy as cp
import numpy as np

from spatial.config import (
    load_config,
)
from spatial.grid import (
    SpatialGrid,
)
from spatial.species import (
    SpeciesRegistry,
)
from spatial.voxel_count_diffusion import (
    VoxelCountDiffusion,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


def build_state():

    config = load_config(
        "configs/default.yaml"
    )


    registry = SpeciesRegistry.from_yaml(
        "tests/data/"
        "voxel_count_species.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    state = VoxelCountState(
        grid=grid,
        registry=registry,
        seed=config.seed,
    )


    ribosome = registry.get(
        "Ribosome"
    )


    timestep_s = (
        config.spatial
        .diffusion_timestep_s(
            ribosome.diffusion_um2_s
        )
    )


    diffusion = VoxelCountDiffusion(
        voxel_count_state=state,
        spatial_config=config.spatial,
        timestep_s=timestep_s,
        seed=config.seed,
    )


    return (
        config,
        registry,
        grid,
        state,
        diffusion,
    )


def test_voxel_count_diffusion_spreads_and_conserves():

    (
        config,
        registry,
        grid,
        state,
        diffusion,
    ) = build_state()


    # ========================================================
    # Replace the random initialization with a controlled
    # pulse of 20,000 ribosomes in one interior voxel.
    # ========================================================

    state.counts.fill(
        0
    )


    state.counts_next.fill(
        0
    )


    source_voxel = None


    for voxel in range(
        grid.cell_voxels
    ):

        neighbors = (
            grid.neighbors_cpu[
                voxel
            ]
        )


        if np.all(
            neighbors
            != voxel
        ):

            source_voxel = voxel

            break


    assert (
        source_voxel
        is not None
    )


    ribosome_id = (
        registry.get(
            "Ribosome"
        ).voxel_count_id
    )


    initial_count = 20000


    state.counts[
        ribosome_id,
        source_voxel
    ] = initial_count


    initial_total = int(
        cp.sum(
            state.counts[
                ribosome_id
            ],
            dtype=cp.uint64,
        ).item()
    )


    assert (
        initial_total
        == initial_count
    )


    # ========================================================
    # One stochastic diffusion step
    # ========================================================

    diffusion.step()


    cp.cuda.Stream.null.synchronize()


    final_total = int(
        cp.sum(
            state.counts[
                ribosome_id
            ],
            dtype=cp.uint64,
        ).item()
    )


    center_count = int(
        state.counts[
            ribosome_id,
            source_voxel
        ].item()
    )


    nonzero_voxels = int(
        cp.count_nonzero(
            state.counts[
                ribosome_id
            ]
        ).item()
    )


    # ========================================================
    # Exact conservation
    # ========================================================

    assert (
        final_total
        == initial_total
    )


    # ========================================================
    # Counts remain integer-valued.
    # ========================================================

    assert (
        state.counts.dtype
        == cp.uint32
    )


    # ========================================================
    # At the configured safe timestep:
    #
    # total hop probability is 0.5.
    #
    # So approximately half should remain in the source voxel.
    # Use a deliberately broad stochastic tolerance.
    # ========================================================

    assert (
        8000
        < center_count
        < 12000
    )


    # ========================================================
    # Molecules actually spread into neighboring voxels.
    # ========================================================

    assert (
        nonzero_voxels
        > 1
    )


    assert (
        diffusion.attempted_hop_count
        > 0
    )


    assert (
        diffusion.actual_hop_count
        > 0
    )


    assert diffusion.validate()


def test_voxel_count_diffusion_long_run_conservation():

    (
        config,
        registry,
        grid,
        state,
        diffusion,
    ) = build_state()


    initial = cp.asnumpy(
        cp.sum(
            state.counts,
            axis=1,
            dtype=cp.uint64,
        )
    )


    diffusion.run_steps(
        100
    )


    cp.cuda.Stream.null.synchronize()


    final = cp.asnumpy(
        cp.sum(
            state.counts,
            axis=1,
            dtype=cp.uint64,
        )
    )


    # ========================================================
    # Integer diffusion should conserve every species exactly.
    #
    # Unlike float32 concentration fields, there should be
    # zero floating-point drift because molecule counts are
    # integers.
    # ========================================================

    assert np.array_equal(
        initial,
        final,
    )


    assert (
        diffusion.actual_hop_count
        > 0
    )


    assert diffusion.validate()
