from spatial.config import load_config
from spatial.grid import SpatialGrid
from spatial.particles import ParticleState
from spatial.species import SpeciesRegistry


def test_particle_state_conserves_initial_particles():

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

    assert state.validate()

    expected = sum(
        species.initial_count or 0
        for species
        in registry.particle_species
    )

    assert (
        state.total_particle_count
        == expected
    )
