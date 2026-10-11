import pytest

from spatial.config import load_config
from spatial.fields import FieldState
from spatial.grid import SpatialGrid
from spatial.species import SpeciesRegistry


def test_field_state_initialization():

    config = load_config(
        "configs/default.yaml"
    )

    registry = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )

    grid = SpatialGrid.from_config(
        config.spatial
    )

    state = FieldState(
        grid=grid,
        registry=registry,
        config=config.fields,
    )

    assert state.validate()

    assert state.values.shape == (
        registry.field_count,
        grid.cell_voxels,
    )


    for species in registry.field_species:

        observed = (
            state.mean_concentration_mM(
                species.name
            )
        )

        expected = (
            species.initial_concentration_mM
            or 0.0
        )

        assert observed == pytest.approx(
            expected,
            rel=1e-6,
        )
