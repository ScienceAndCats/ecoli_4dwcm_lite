import cupy as cp
import numpy as np
import pytest

from spatial.config import load_config
from spatial.field_diffusion import FieldDiffusion
from spatial.fields import FieldState
from spatial.grid import SpatialGrid
from spatial.species import SpeciesRegistry


def test_field_diffusion_spreads_and_conserves():

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


    fastest_diffusion = (
        registry.fastest_diffusion_um2_s
    )

    base_dt = (
        config.spatial.diffusion_timestep_s(
            fastest_diffusion
        )
    )


    diffusion = FieldDiffusion(
        field_state=state,
        spatial_config=config.spatial,
        timestep_s=base_dt,
    )


    assert registry.field_count >= 1


    # ========================================================
    # Replace normal uniform starting state with a test pulse.
    # ========================================================

    state.values.fill(0)
    state.values_next.fill(0)


    # Find an interior voxel with six true neighbors.

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
            neighbors != voxel
        ):

            source_voxel = voxel
            break


    assert source_voxel is not None


    field_id = 0

    state.values[
        field_id,
        source_voxel
    ] = 100.0


    initial_total = float(
        cp.sum(
            state.values[
                field_id
            ]
        ).item()
    )

    initial_center = float(
        state.values[
            field_id,
            source_voxel
        ].item()
    )


    # ========================================================
    # One diffusion step
    # ========================================================

    diffusion.step()

    cp.cuda.Stream.null.synchronize()


    final_total = float(
        cp.sum(
            state.values[
                field_id
            ]
        ).item()
    )

    final_center = float(
        state.values[
            field_id,
            source_voxel
        ].item()
    )


    nonzero_voxels = int(
        cp.count_nonzero(
            state.values[
                field_id
            ]
        ).item()
    )


    # ========================================================
    # Tests
    # ========================================================

    # Mass/concentration sum should be conserved.

    assert final_total == pytest.approx(
        initial_total,
        rel=1e-6,
        abs=1e-5,
    )


    # Concentration should leave the original voxel.

    assert final_center < initial_center


    # It should have spread beyond that voxel.

    assert nonzero_voxels > 1


    # Concentration should never become negative.

    assert bool(
        cp.all(
            state.values >= 0
        ).item()
    )


    assert diffusion.validate()
