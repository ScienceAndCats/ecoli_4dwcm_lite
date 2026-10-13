import cupy as cp
import numpy as np

from spatial.config import load_config
from spatial.crowding import CrowdingState
from spatial.effective_volumes import (
    EffectiveVolumeRegistry,
)
from spatial.grid import SpatialGrid
from spatial.particle_diffusion import (
    ParticleDiffusion,
)
from spatial.particles import ParticleState
from spatial.species import SpeciesRegistry
from spatial.voxel_count_diffusion import (
    VoxelCountDiffusion,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


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


def build_voxel_count_context():

    config = load_config(
        "configs/default.yaml"
    )


    registry = SpeciesRegistry.from_yaml(
        "tests/data/"
        "voxel_count_species.yaml"
    )


    volumes = EffectiveVolumeRegistry.from_yaml(
        path=(
            "tests/data/"
            "effective_volumes.yaml"
        ),

        species_registry=registry,
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    state = VoxelCountState(
        grid=grid,
        registry=registry,
        seed=config.seed,
    )


    return (
        config,
        registry,
        volumes,
        grid,
        state,
    )


def test_voxel_count_cannot_enter_if_molecule_does_not_fit():

    (
        config,
        registry,
        volumes,
        grid,
        state,
    ) = build_voxel_count_context()


    state.counts.fill(
        0
    )

    state.counts_next.fill(
        0
    )


    source = find_interior_voxel(
        grid
    )


    destination = int(
        grid.neighbors_cpu[
            source,
            0
        ]
    )


    ribosome = registry.get(
        "Ribosome"
    )


    ribosome_id = (
        ribosome.voxel_count_id
    )


    # Test volume configuration:
    #
    # Ribosome = 8,000 nm^3

    assert (
        volumes.species_volume_nm3(
            "Ribosome"
        )
        == 8000.0
    )


    state.counts[
        ribosome_id,
        source
    ] = 20000


    # ========================================================
    # Destination:
    #
    # voxel capacity = 125,000
    # occupied       = 120,000
    # free           =   5,000
    #
    # Ribosome       =   8,000
    #
    # Therefore ZERO ribosomes can physically enter.
    # ========================================================

    crowding = CrowdingState(
        grid=grid,

        # Nearly eliminates the soft probability penalty so
        # this test isolates hard capacity.
        movement_exponent=0.000001,
    )


    crowding.add_static_volume(
        voxel_ids=[
            destination
        ],

        volumes_nm3=120000.0,
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            ribosome.diffusion_um2_s
        )
    )


    diffusion = VoxelCountDiffusion(
        voxel_count_state=state,

        spatial_config=config.spatial,

        timestep_s=timestep,

        seed=config.seed,

        crowding_state=crowding,

        effective_volumes=volumes,
    )


    diffusion.step()


    cp.cuda.Stream.null.synchronize()


    assert (
        int(
            state.counts[
                ribosome_id,
                destination
            ].item()
        )
        == 0
    )


    assert (
        diffusion.capacity_rejection_count
        > 0
    )


    assert (
        int(
            cp.sum(
                state.counts[
                    ribosome_id
                ],
                dtype=cp.uint64,
            ).item()
        )
        == 20000
    )


def test_only_one_ribosome_enters_when_exactly_one_fits():

    (
        config,
        registry,
        volumes,
        grid,
        state,
    ) = build_voxel_count_context()


    state.counts.fill(
        0
    )

    state.counts_next.fill(
        0
    )


    source = find_interior_voxel(
        grid
    )


    destination = int(
        grid.neighbors_cpu[
            source,
            0
        ]
    )


    ribosome = registry.get(
        "Ribosome"
    )


    ribosome_id = (
        ribosome.voxel_count_id
    )


    state.counts[
        ribosome_id,
        source
    ] = 20000


    crowding = CrowdingState(
        grid=grid,
        movement_exponent=0.000001,
    )


    # ========================================================
    # Exactly 8,000 nm^3 remains.
    #
    # Exactly ONE 8,000-nm^3 ribosome should therefore be
    # allowed into this voxel during the diffusion step.
    # ========================================================

    crowding.add_static_volume(
        voxel_ids=[
            destination
        ],

        volumes_nm3=117000.0,
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            ribosome.diffusion_um2_s
        )
    )


    diffusion = VoxelCountDiffusion(
        voxel_count_state=state,

        spatial_config=config.spatial,

        timestep_s=timestep,

        seed=config.seed,

        crowding_state=crowding,

        effective_volumes=volumes,
    )


    diffusion.step()


    cp.cuda.Stream.null.synchronize()


    entered = int(
        state.counts[
            ribosome_id,
            destination
        ].item()
    )


    assert (
        entered == 1
    )


    assert (
        diffusion.capacity_rejection_count
        > 0
    )


    # The GPU reservation itself may never exceed the
    # remaining physical capacity.

    reserved = float(
        diffusion.reserved_incoming_nm3[
            destination
        ].item()
    )


    assert (
        reserved <= 8000.0
    )


    assert (
        int(
            cp.sum(
                state.counts[
                    ribosome_id
                ],
                dtype=cp.uint64,
            ).item()
        )
        == 20000
    )


def test_explicit_particle_cannot_enter_if_it_does_not_fit():

    config = load_config(
        "configs/default.yaml"
    )


    registry = SpeciesRegistry.from_yaml(
        "tests/data/"
        "voxel_count_species.yaml"
    )


    volumes = EffectiveVolumeRegistry.from_yaml(
        path=(
            "tests/data/"
            "effective_volumes.yaml"
        ),

        species_registry=registry,
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


    # Controlled state: exactly one TF.

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


    tf = registry.get(
        "example_TF"
    )


    tf_id = (
        tf.particle_id
    )


    assert (
        volumes.species_volume_nm3(
            "example_TF"
        )
        == 100.0
    )


    state.primary[
        source,
        0
    ] = tf_id


    state.primary_occupancy[
        source
    ] = 1


    crowding = CrowdingState(
        grid=grid,
        movement_exponent=0.000001,
    )


    neighbors = (
        grid.neighbors_cpu[
            source
        ]
    )


    # ========================================================
    # Leave only 50 nm^3 free in every neighboring voxel.
    #
    # TF effective volume = 100 nm^3.
    #
    # Therefore every attempted movement out of the source
    # must fail the hard capacity test.
    # ========================================================

    crowding.add_static_volume(
        voxel_ids=neighbors,

        volumes_nm3=(
            crowding.voxel_volume_nm3
            - 50.0
        ),
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            tf.diffusion_um2_s
        )
    )


    diffusion = ParticleDiffusion(
        particle_state=state,

        spatial_config=config.spatial,

        base_dt_s=timestep,

        seed=config.seed,

        crowding_state=crowding,

        effective_volumes=volumes,
    )


    diffusion.run_steps(
        200
    )


    cp.cuda.Stream.null.synchronize()


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
        diffusion.capacity_rejection_count
        > 0
    )


    assert diffusion.validate()
