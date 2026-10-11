import cupy as cp
import numpy as np

from spatial.config import load_config
from spatial.grid import SpatialGrid
from spatial.particle_diffusion import ParticleDiffusion
from spatial.particles import ParticleState
from spatial.species import SpeciesRegistry


def voxel_occupancy(
    state,
):

    return cp.asnumpy(
        state.primary_occupancy
        + state.spill_occupancy
    )


def species_counts(
    state,
):

    particle_count = (
        state.registry.particle_count
    )

    primary_counts = cp.bincount(
        state.primary.reshape(-1),
        minlength=particle_count + 1,
    )

    spill_counts = cp.bincount(
        state.spill.reshape(-1),
        minlength=particle_count + 1,
    )

    return cp.asnumpy(
        primary_counts
        + spill_counts
    )


def test_particle_diffusion_moves_and_conserves():

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


    fastest_diffusion = (
        registry.fastest_diffusion_um2_s
    )

    base_dt = (
        config.spatial.diffusion_timestep_s(
            fastest_diffusion
        )
    )


    diffusion = ParticleDiffusion(
        particle_state=state,
        spatial_config=config.spatial,
        base_dt_s=base_dt,
        seed=config.seed,
    )


    initial_total = (
        state.total_particle_count
    )

    initial_species = species_counts(
        state
    )

    initial_occupancy = voxel_occupancy(
        state
    )


    diffusion.run_steps(
        100
    )

    cp.cuda.Stream.null.synchronize()


    final_total = (
        state.total_particle_count
    )

    final_species = species_counts(
        state
    )

    final_occupancy = voxel_occupancy(
        state
    )


    # ========================================================
    # Conservation
    # ========================================================

    assert final_total == initial_total

    assert np.array_equal(
        initial_species,
        final_species,
    )

    assert (
        diffusion.hard_overflow_count
        == 0
    )


    # ========================================================
    # Confirm that the spatial distribution changed.
    # ========================================================

    changed_voxels = int(
        np.count_nonzero(
            initial_occupancy
            != final_occupancy
        )
    )

    assert changed_voxels > 0


    assert diffusion.validate()
