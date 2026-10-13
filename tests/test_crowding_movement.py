import cupy as cp
import numpy as np
import pytest

from spatial.config import load_config
from spatial.crowding import CrowdingState
from spatial.grid import SpatialGrid
from spatial.particle_diffusion import ParticleDiffusion
from spatial.particles import ParticleState
from spatial.species import SpeciesRegistry
from spatial.voxel_count_diffusion import (
    VoxelCountDiffusion,
)
from spatial.voxel_counts import VoxelCountState


def find_interior_voxel(
    grid,
):

    for voxel in range(
        grid.cell_voxels
    ):

        neighbors = (
            grid.neighbors_cpu[
                voxel
            ]
        )


        if np.all(
            neighbors != voxel
        ):

            return voxel


    raise RuntimeError(
        "No interior voxel found."
    )


def test_accessibility_tracks_free_volume():

    config = load_config(
        "configs/default.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    voxels = [
        10,
        20,
        30,
    ]


    crowding.add_dynamic_volume(
        voxel_ids=voxels,

        volumes_nm3=[
            0.0,
            31250.0,
            93750.0,
        ],
    )


    crowding.refresh_movement_accessibility()


    observed = cp.asnumpy(
        crowding.movement_accessibility[
            voxels
        ]
    )


    assert observed[
        0
    ] == pytest.approx(
        1.00
    )


    assert observed[
        1
    ] == pytest.approx(
        0.75
    )


    assert observed[
        2
    ] == pytest.approx(
        0.25
    )


def test_voxel_count_diffusion_avoids_crowded_destination():

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


    state.counts.fill(
        0
    )


    state.counts_next.fill(
        0
    )


    source = find_interior_voxel(
        grid
    )


    neighbors = (
        grid.neighbors_cpu[
            source
        ]
    )


    crowded_neighbor = int(
        neighbors[
            0
        ]
    )


    open_neighbor = int(
        neighbors[
            1
        ]
    )


    ribosome = registry.get(
        "Ribosome"
    )


    ribosome_id = (
        ribosome.voxel_count_id
    )


    initial_count = 20000


    state.counts[
        ribosome_id,
        source
    ] = initial_count


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    # 90% full.
    #
    # With exponent=1 this destination has only 10%
    # accessibility.

    crowding.add_dynamic_volume(
        voxel_ids=[
            crowded_neighbor
        ],

        volumes_nm3=(
            crowding.voxel_volume_nm3
            * 0.90
        ),
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            ribosome.diffusion_um2_s
        )
    )


    diffusion = VoxelCountDiffusion(
        voxel_count_state=state,

        spatial_config=(
            config.spatial
        ),

        timestep_s=timestep,

        seed=config.seed,

        crowding_state=crowding,
    )


    diffusion.step()


    cp.cuda.Stream.null.synchronize()


    crowded_count = int(
        state.counts[
            ribosome_id,
            crowded_neighbor
        ].item()
    )


    open_count = int(
        state.counts[
            ribosome_id,
            open_neighbor
        ].item()
    )


    final_total = int(
        cp.sum(
            state.counts[
                ribosome_id
            ],
            dtype=cp.uint64,
        ).item()
    )


    # Exact molecule conservation.

    assert (
        final_total
        == initial_count
    )


    # The 90%-occupied voxel should receive far fewer
    # ribosomes than an equivalent empty neighbor.

    assert (
        crowded_count
        < open_count
        * 0.25
    )


    assert (
        diffusion.crowding_rejection_count
        > 0
    )


    assert diffusion.validate()


def test_particles_cannot_enter_zero_accessibility_neighbors():

    config = load_config(
        "configs/default.yaml"
    )


    registry = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    state = ParticleState(
        grid=grid,
        registry=registry,
        config=config.particles,
        seed=config.seed,
    )


    # Replace normal initialization with one controlled
    # explicit TF particle.

    state.primary.fill(
        0
    )

    state.spill.fill(
        0
    )

    state.primary_occupancy.fill(
        0
    )

    state.spill_occupancy.fill(
        0
    )


    source = find_interior_voxel(
        grid
    )


    tf_id = registry.get(
        "example_TF"
    ).particle_id


    state.primary[
        source,
        0
    ] = tf_id


    state.primary_occupancy[
        source
    ] = 1


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    # Make all six neighboring voxels completely inaccessible.

    neighbors = (
        grid.neighbors_cpu[
            source
        ]
    )


    crowding.add_dynamic_volume(
        voxel_ids=neighbors,

        volumes_nm3=(
            crowding.voxel_volume_nm3
        ),
    )


    fastest = max(
        species.diffusion_um2_s

        for species
        in registry.particle_species
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            fastest
        )
    )


    diffusion = ParticleDiffusion(
        particle_state=state,

        spatial_config=(
            config.spatial
        ),

        base_dt_s=timestep,

        seed=config.seed,

        crowding_state=crowding,
    )


    diffusion.run_steps(
        200
    )


    cp.cuda.Stream.null.synchronize()


    # The only particle must remain in its original voxel.

    assert (
        state.total_particle_count
        == 1
    )


    assert (
        int(
            state.primary_occupancy[
                source
            ].item()
        )
        == 1
    )


    assert (
        int(
            state.primary[
                source,
                0
            ].item()
        )
        == tf_id
    )


    assert (
        diffusion.crowding_rejection_count
        > 0
    )


    assert diffusion.validate()
