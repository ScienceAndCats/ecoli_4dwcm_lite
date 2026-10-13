import cupy as cp

from spatial.config import (
    load_config,
)
from spatial.grid import (
    SpatialGrid,
)
from spatial.species import (
    Representation,
    SpeciesRegistry,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


def test_voxel_count_species_registry():

    registry = (
        SpeciesRegistry.from_yaml(
            "tests/data/"
            "voxel_count_species.yaml"
        )
    )


    assert (
        registry.total_species
        == 3
    )


    assert (
        registry.particle_count
        == 1
    )


    assert (
        registry.field_count
        == 1
    )


    assert (
        registry.voxel_count_count
        == 1
    )


    ribosome = (
        registry.get(
            "Ribosome"
        )
    )


    assert (
        ribosome.representation
        is Representation.VOXEL_COUNT
    )


    assert (
        ribosome.voxel_count_id
        == 0
    )


    assert (
        ribosome.particle_id
        is None
    )


    assert (
        ribosome.field_id
        is None
    )


def test_voxel_count_state_initialization():

    config = load_config(
        "configs/default.yaml"
    )


    registry = (
        SpeciesRegistry.from_yaml(
            "tests/data/"
            "voxel_count_species.yaml"
        )
    )


    grid = (
        SpatialGrid.from_config(
            config.spatial
        )
    )


    state = (
        VoxelCountState(
            grid=grid,
            registry=registry,
            seed=config.seed,
        )
    )


    assert state.validate()


    assert (
        state.counts.shape
        ==
        (
            1,
            grid.cell_voxels,
        )
    )


    assert (
        state.counts.dtype
        == cp.uint32
    )


    assert (
        state.total_count(
            "Ribosome"
        )
        == 20000
    )


    # The initial population should be distributed over
    # many voxels, not stored as one whole-cell pool.

    occupied_voxels = int(
        cp.count_nonzero(
            state.get_counts(
                "Ribosome"
            )
        ).item()
    )


    assert (
        occupied_voxels
        > 1
    )


    expected_mean = (
        20000
        / grid.cell_voxels
    )


    assert abs(
        state.mean_count_per_voxel(
            "Ribosome"
        )
        - expected_mean
    ) < 1e-6


    # The diffusion buffer starts empty.

    assert (
        int(
            cp.sum(
                state.counts_next
            ).item()
        )
        == 0
    )
