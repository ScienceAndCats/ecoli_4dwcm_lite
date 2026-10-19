import cupy as cp
import numpy as np
import pytest

from spatial.config import load_config
from spatial.crowding import CrowdingState
from spatial.crowding_sync import (
    DynamicCrowdingSynchronizer,
)
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


def build_controlled_context():

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


    particles = ParticleState(
        grid=grid,
        registry=registry,
        config=config.particles,
        seed=config.seed,
    )


    voxel_counts = VoxelCountState(
        grid=grid,
        registry=registry,
        seed=config.seed,
    )


    # ========================================================
    # Clear normal randomized initialization.
    # ========================================================

    particles.primary.fill(
        0
    )

    particles.spill.fill(
        0
    )

    particles.primary_occupancy.fill(
        0
    )

    particles.spill_occupancy.fill(
        0
    )


    voxel_counts.counts.fill(
        0
    )

    voxel_counts.counts_next.fill(
        0
    )


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    return (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    )


def test_sync_rebuilds_particle_and_voxel_count_volume():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    tf_id = registry.get(
        "example_TF"
    ).particle_id


    ribosome_id = registry.get(
        "Ribosome"
    ).voxel_count_id


    # ========================================================
    # Voxel 10:
    #
    # 2 TF       × 100   =    200
    # 3 ribosome × 8000  = 24,000
    #
    # total               = 24,200 nm^3
    # ========================================================

    particles.primary[
        10,
        0:2
    ] = tf_id


    particles.primary_occupancy[
        10
    ] = 2


    voxel_counts.counts[
        ribosome_id,
        10
    ] = 3


    # ========================================================
    # Voxel 20:
    #
    # 1 TF       × 100   =    100
    # 5 ribosome × 8000  = 40,000
    #
    # total               = 40,100 nm^3
    # ========================================================

    particles.primary[
        20,
        0
    ] = tf_id


    particles.primary_occupancy[
        20
    ] = 1


    voxel_counts.counts[
        ribosome_id,
        20
    ] = 5


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    assert (
        float(
            crowding.dynamic_volume_nm3[
                10
            ].item()
        )
        == pytest.approx(
            24200.0
        )
    )


    assert (
        float(
            crowding.dynamic_volume_nm3[
                20
            ].item()
        )
        == pytest.approx(
            40100.0
        )
    )


    assert (
        sync.dynamic_volume_nm3
        == pytest.approx(
            64300.0
        )
    )


    assert sync.validate()


def test_rebuild_tracks_molecule_movement_without_volume_loss():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    ribosome_id = registry.get(
        "Ribosome"
    ).voxel_count_id


    voxel_counts.counts[
        ribosome_id,
        10
    ] = 3


    voxel_counts.counts[
        ribosome_id,
        20
    ] = 5


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    before_total = (
        sync.dynamic_volume_nm3
    )


    # Move exactly one ribosome:
    #
    # voxel 10 → voxel 20

    voxel_counts.counts[
        ribosome_id,
        10
    ] -= 1


    voxel_counts.counts[
        ribosome_id,
        20
    ] += 1


    sync.rebuild_dynamic()


    assert (
        float(
            crowding.dynamic_volume_nm3[
                10
            ].item()
        )
        == pytest.approx(
            16000.0
        )
    )


    assert (
        float(
            crowding.dynamic_volume_nm3[
                20
            ].item()
        )
        == pytest.approx(
            48000.0
        )
    )


    # Molecules moved, but none were created or destroyed.

    assert (
        sync.dynamic_volume_nm3
        == pytest.approx(
            before_total
        )
    )


    assert sync.validate()


def test_extra_volume_provider_is_included():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    extra = cp.zeros(
        grid.cell_voxels,
        dtype=cp.float32,
    )


    extra[
        50
    ] = 1234.0


    sync.register_volume_provider(
        "future_mrna_example",

        lambda: extra,
    )


    assert (
        float(
            crowding.dynamic_volume_nm3[
                50
            ].item()
        )
        == pytest.approx(
            1234.0
        )
    )


    assert (
        sync.provider_count
        == 1
    )


    assert sync.validate()


def test_voxel_count_diffusion_automatically_resynchronizes():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    source = find_interior_voxel(
        grid
    )


    ribosome = registry.get(
        "Ribosome"
    )


    ribosome_id = (
        ribosome.voxel_count_id
    )


    # 10 test ribosomes:
    #
    # 10 × 8000 = 80,000 nm^3
    #
    # This fits inside one 125,000-nm^3 voxel.

    voxel_counts.counts[
        ribosome_id,
        source
    ] = 10


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            ribosome.diffusion_um2_s
        )
    )


    diffusion = VoxelCountDiffusion(
        voxel_count_state=voxel_counts,

        spatial_config=config.spatial,

        timestep_s=timestep,

        seed=config.seed,

        crowding_synchronizer=sync,
    )


    diffusion.run_steps(
        20
    )


    cp.cuda.Stream.null.synchronize()


    expected = (
        voxel_counts.counts[
            ribosome_id
        ].astype(
            cp.float32
        )
        * 8000.0
    )


    cp.testing.assert_allclose(
        crowding.dynamic_volume_nm3,
        expected,
        rtol=1e-6,
        atol=1e-3,
    )


    assert (
        voxel_counts.total_count(
            "Ribosome"
        )
        == 10
    )


    assert (
        sync.dynamic_volume_nm3
        == pytest.approx(
            80000.0
        )
    )


    assert (
        sync.rebuild_count
        > 1
    )


    assert diffusion.validate()


def test_particle_diffusion_automatically_resynchronizes():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    source = find_interior_voxel(
        grid
    )


    tf = registry.get(
        "example_TF"
    )


    tf_id = (
        tf.particle_id
    )


    particles.primary[
        source,
        0
    ] = tf_id


    particles.primary_occupancy[
        source
    ] = 1


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    timestep = (
        config.spatial
        .diffusion_timestep_s(
            tf.diffusion_um2_s
        )
    )


    diffusion = ParticleDiffusion(
        particle_state=particles,

        spatial_config=config.spatial,

        base_dt_s=timestep,

        seed=config.seed,

        crowding_synchronizer=sync,
    )


    diffusion.run_steps(
        50
    )


    cp.cuda.Stream.null.synchronize()


    expected = (
        cp.sum(
            sync.particle_volume_lut_nm3[
                particles.primary
            ],

            axis=1,
            dtype=cp.float32,
        )
    )


    if particles.spill_slots > 0:

        expected += cp.sum(
            sync.particle_volume_lut_nm3[
                particles.spill
            ],

            axis=1,
            dtype=cp.float32,
        )


    cp.testing.assert_allclose(
        crowding.dynamic_volume_nm3,
        expected,
        rtol=1e-6,
        atol=1e-3,
    )


    assert (
        particles.total_particle_count
        == 1
    )


    assert (
        sync.dynamic_volume_nm3
        == pytest.approx(
            100.0
        )
    )


    assert (
        sync.rebuild_count
        > 1
    )


    assert diffusion.validate()


def test_mixed_diffusion_conserves_dynamic_volume():

    (
        config,
        registry,
        volumes,
        grid,
        particles,
        voxel_counts,
        crowding,
    ) = build_controlled_context()


    source = find_interior_voxel(
        grid
    )


    tf = registry.get(
        "example_TF"
    )


    ribosome = registry.get(
        "Ribosome"
    )


    particles.primary[
        source,
        0
    ] = tf.particle_id


    particles.primary_occupancy[
        source
    ] = 1


    voxel_counts.counts[
        ribosome.voxel_count_id,
        source
    ] = 10


    sync = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    initial_dynamic_volume = (
        sync.dynamic_volume_nm3
    )


    assert (
        initial_dynamic_volume
        == pytest.approx(
            80100.0
        )
    )


    particle_diffusion = ParticleDiffusion(
        particle_state=particles,

        spatial_config=config.spatial,

        base_dt_s=(
            config.spatial
            .diffusion_timestep_s(
                tf.diffusion_um2_s
            )
        ),

        seed=config.seed,

        crowding_synchronizer=sync,
    )


    voxel_count_diffusion = VoxelCountDiffusion(
        voxel_count_state=voxel_counts,

        spatial_config=config.spatial,

        timestep_s=(
            config.spatial
            .diffusion_timestep_s(
                ribosome.diffusion_um2_s
            )
        ),

        seed=(
            config.seed
            + 1
        ),

        crowding_synchronizer=sync,
    )


    for _ in range(
        50
    ):

        particle_diffusion.step()

        voxel_count_diffusion.step()


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Number of physical objects is unchanged.
    # ========================================================

    assert (
        particles.total_particle_count
        == 1
    )


    assert (
        voxel_counts.total_count(
            "Ribosome"
        )
        == 10
    )


    # ========================================================
    # Therefore total occupied molecular volume must also be
    # unchanged.
    # ========================================================

    assert (
        sync.dynamic_volume_nm3
        == pytest.approx(
            initial_dynamic_volume,
            rel=1e-6,
        )
    )


    # ========================================================
    # No voxel may exceed its physical volume.
    # ========================================================

    assert (
        crowding.max_occupied_fraction
        <= 1.0
    )


    assert sync.validate()

    assert particle_diffusion.validate()

    assert voxel_count_diffusion.validate()
